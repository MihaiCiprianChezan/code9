import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time
import os
import gc
import argparse
import json
from optimum.bettertransformer import BetterTransformer


# Constants for available models
class Models:
    QWEN_0_5B = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    QWEN_1_5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    QWEN_3B = "Qwen/Qwen2.5-Coder-3B-Instruct"
    STAR_CODER_3B = "bigcode/starcoder2-3b"

    DEFAULT = QWEN_0_5B

    @classmethod
    def get_cache_dir(cls, model_name):
        """Generate cache directory path for a model"""
        return os.path.join("./model_cache/", model_name.split("/")[-1].replace(".", "_"))


class PromptTemplates:
    """Collection of system prompts for different analysis tasks"""

    CHUNK_ANALYSIS = (
        "You are a log analysis expert. Analyze this log file segment. "
        "Look for errors, critical issues, warnings, unusual patterns, and security concerns. "
        "Give only a summary of the most important findings. Avoid redundancy. "
        "Be concise but thorough. "
        "Focus on the most important findings in this segment and be short and brief in explanations. "
        "Prioritise and highlight by criticality. Discard less critical or less important findings."
    )

    SUMMARIZATION = (
        "You are a log analysis expert. I will provide you with analyses of different segments "
        "of a large log file. Synthesize these analyses into one unified and comprehensive summary, "
        "not as individual chunks or segments. "
        "Identify all patterns across all segments into one analysis, avoid redundancy, "
        "and focus on the most important issues. Categorize findings by severity (Critical, Error, Warning). "
        "Avoid Info type and less critical findings. "
        "Highlight any root causes and security concerns."
    )


class ModelManager:
    """Handles model loading, optimization, and cleanup"""

    def __init__(self, model_name, cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self):
        """Set up the environment and configuration for model loading"""
        # Set environment variables for optimization
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))

        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        print(f"Using device: {self.device}")

    def print_cuda_info(self):
        """Print CUDA availability information"""
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")

    def print_device_map(self):
        """Print where each model component is placed"""
        if not self.model:
            return

        print("\nModel device map:")
        for name, module in self.model.named_modules():
            if hasattr(module, 'device'):
                print(f"{name}: {module.device}")
            elif hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                print(f"{name}: {module.weight.device}")

    def load_model(self):
        """Load and optimize the model for inference"""
        print(f"Loading model {self.model_name}...")

        # Free up memory before loading model
        self._free_memory()
        self.print_cuda_info()

        self.model = self._load_fresh_model()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        # Ensure tokenizer has a pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optimize model for inference
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # Warm up the model
        self._warm_up_model()

    def _try_load_cached_model(self):
        """Try to load the model from cache if available"""
        if not self.cache_dir:
            return False

        model_path = os.path.join(self.cache_dir, "log_analyzer_model")
        if os.path.exists(model_path):
            print("Loading from cached model...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                return True
            except Exception as e:
                print(f"Failed to load from cache: {e}")
                return False
        return False

    def _cache_model(self):
        """Cache model for future use"""
        if self.cache_dir and self.model:
            os.makedirs(self.cache_dir, exist_ok=True)
            model_path = os.path.join(self.cache_dir, "log_analyzer_model")
            self.model.save_pretrained(model_path)
            print(f"Model cached at {model_path}")

    def _load_fresh_model(self):
        """Load model with optimizations"""
        try:
            # First attempt to load with Flash Attention 2
            print("Attempting to load model with Flash Attention...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation="flash_attention_2",
            )
            print("Successfully loaded model with Flash Attention!")
            return model
        except Exception as e:
            print(f"Failed to load with Flash Attention: {e}")
            print("Falling back to standard attention implementation...")
            # Fall back to standard implementation
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_cache=True,
            )

            if torch.cuda.is_available():
                model = model.half().to(self.device)
            else:
                model = model.to(self.device)

            return BetterTransformer.transform(model)

    def _warm_up_model(self):
        """Warm up the model to initialize CUDA kernels"""
        print("Warming up model...")
        dummy_input = "Hello, this is a warm-up."
        inputs = self.tokenizer([dummy_input], return_tensors="pt").to(self.model.device)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = self.model(**{k: v for k, v in inputs.items() if k != 'token_type_ids'}, max_length=1)

    def _free_memory(self):
        """Free up memory"""
        gc.collect()
        torch.cuda.empty_cache()

    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self._free_memory()


class TextGenerator:
    """Handles text generation using the loaded model"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def generate(self, messages, max_tokens=512, temperature=0.7, do_sample=True, show_output=True):
        """Generate text from messages using the model"""
        if not self.model_manager.model or not self.model_manager.tokenizer:
            raise ValueError("Model not loaded")

        input_text = self.model_manager.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.model_manager.tokenizer(
            [input_text],
            return_tensors="pt"
        ).to(self.model_manager.model.device)

        streamer = None
        if show_output:
            streamer = TextStreamer(
                self.model_manager.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )

        start_time = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.model_manager.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
                top_k=10 if do_sample else 50,
                use_cache=True,
                streamer=streamer,
                num_beams=1,
                pad_token_id=self.model_manager.tokenizer.pad_token_id,
            )
        end_time = time.time()

        generated_text = self.model_manager.tokenizer.decode(
            output[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "processing_time": end_time - start_time
        }


class LogChunker:
    """Handles splitting log files into processable chunks"""

    def __init__(self, tokenizer, max_chunk_tokens=10000, overlap_tokens=200):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def read_file_into_chunks(self, file_path):
        """
        Read the specified file and split it into chunks based on token counts.
        """
        print(f"Reading file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {e}")

        total_lines = len(lines)
        chunks = []
        chunk_lines = []
        current_token_count = 0
        start_line = 1  # record the file line number of the first line in the current chunk
        line_index = 0

        # Pre-compute token counts for all lines - do tokenization only once
        print("Pre-calculating token counts...")
        token_counts = {}
        for i, line in enumerate(lines):
            stripped_line = line.rstrip("\n")
            token_counts[i] = len(self.tokenizer.encode(stripped_line, add_special_tokens=False))
            lines[i] = stripped_line  # Store the stripped line to avoid doing it repeatedly

        print("Calculating chunks...")
        while line_index < total_lines:
            line = lines[line_index]
            token_count_line = token_counts[line_index]

            # Check if adding this line would exceed the token limit
            if current_token_count + token_count_line > self.max_chunk_tokens and chunk_lines:
                self._add_chunk(chunks, chunk_lines, start_line)

                # Implement overlap: reuse some lines from the end
                overlap_line_count = min(3, len(chunk_lines))  # Use at most 3 lines or all available lines
                overlap_lines = chunk_lines[-overlap_line_count:]

                # Reset with overlap lines
                chunk_lines = overlap_lines.copy()
                # Use cached token counts instead of recalculating
                current_token_count = sum(token_counts[line_index - len(overlap_lines) + i]
                                          for i in range(len(overlap_lines)))
                start_line = line_index - len(chunk_lines) + 1
            else:
                chunk_lines.append(line)
                current_token_count += token_count_line
                line_index += 1

        # Add the final chunk if there are remaining lines
        if chunk_lines:
            self._add_chunk(chunks, chunk_lines, start_line)

        # Update each chunk with the total number of chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        # Use pre-calculated token counts for statistics
        estimated_tokens = sum(token_counts.values())
        print(f"Estimated file size: ~{estimated_tokens} tokens")
        print(f"File split into {total_chunks} chunks.")

        return chunks

    def _add_chunk(self, chunks, chunk_lines, start_line):
        """Add a new chunk to the chunks list"""
        chunk_text = "\n".join(chunk_lines)
        chunks.append({
            "text": chunk_text,
            "start_line": start_line,
            "chunk_number": len(chunks) + 1,
            "total_chunks": None  # to be updated later
        })


class IssueDetector:
    """Detects issues in log chunks based on keywords"""

    @staticmethod
    def detect_issues(chunk):
        """Scan the chunk text for possible issues based on keywords"""
        detected_issues = []
        for offset, line in enumerate(chunk["text"].split("\n")):
            if any(keyword in line for keyword in ["ERROR", "Exception", "FATAL", "CRITICAL", "FAILURE"]):
                detected_issues.append({
                    "line": chunk["start_line"] + offset,
                    "description": line.strip()
                })
        return detected_issues


class LogAnalyzer:
    """Main class for analyzing log files"""

    def __init__(self, model_name=Models.DEFAULT, cache_dir=None):
        """Initialize the log analyzer with model configuration"""
        self.model_manager = ModelManager(model_name, cache_dir)
        self.model_manager.initialize()
        self.text_generator = None
        self.chunker = None

        # Analysis configuration
        self.max_chunk_tokens = 10000
        self.overlap_tokens = 200
        self.max_analysis_tokens = 3000

    def _ensure_model_loaded(self):
        """Ensure model is loaded before proceeding"""
        if not self.model_manager.model:
            self.model_manager.load_model()
            self.text_generator = TextGenerator(self.model_manager)
            self.chunker = LogChunker(
                self.model_manager.tokenizer,
                max_chunk_tokens=self.max_chunk_tokens,
                overlap_tokens=self.overlap_tokens
            )

    def analyze_chunk(self, chunk):
        """Analyze a single chunk of the log file"""
        start_time = time.time()
        chunk_number = chunk["chunk_number"]
        total_chunks = chunk["total_chunks"]
        print(f"\nAnalyzing chunk {chunk_number}/{total_chunks} (starting at line {chunk['start_line']})")

        prompt = (
            f"This is chunk {chunk_number} of {total_chunks} from a log file. Analyze it for important issues:\n\n"
            f"{chunk['text']}"
        )

        messages = [
            {"role": "system", "content": PromptTemplates.CHUNK_ANALYSIS},
            {"role": "user", "content": prompt}
        ]

        result = self.text_generator.generate(
            messages=messages,
            max_tokens=self.max_analysis_tokens,
            temperature=0.8
        )

        # Detect issues in the chunk
        detected_issues = IssueDetector.detect_issues(chunk)

        end_time = time.time()
        diff = round(end_time - start_time, 2)
        print(f"Chunk analysis completed in {diff} seconds.")
        return {
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "analysis": result["text"],
            "processing_time": result["processing_time"],
            "detected_issues": detected_issues
        }

    def synthesize_analyses(self, analyses):
        """Synthesize individual chunk analyses into a comprehensive summary"""
        print("\nSynthesizing all analyses into a comprehensive report...")

        combined_analyses = self._prepare_combined_analyses(analyses)
        all_detected_issues = self._collect_all_issues(analyses)

        prompt = (
            f"Here are analyses of different segments of a log file. "
            f"Synthesize them into one comprehensive report:\n\n{combined_analyses}"
        )

        messages = [
            {"role": "system", "content": PromptTemplates.SUMMARIZATION},
            {"role": "user", "content": prompt}
        ]

        print("\n--- Generating comprehensive analysis ---\n")
        result = self.text_generator.generate(
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )

        return {
            "final_analysis": result["text"],
            "chunk_count": len(analyses),
            "processing_time": result["processing_time"],
            "analyses": analyses,
            "detected_issues_combined": all_detected_issues
        }

    def _prepare_combined_analyses(self, analyses):
        """Prepare the combined analyses text, handling token limits"""
        combined_analyses = ""
        for analysis in analyses:
            combined_analyses += f"\n--- ANALYSIS OF CHUNK {analysis['chunk_number']}/{analysis['total_chunks']} ---\n"
            combined_analyses += analysis["analysis"]

            if analysis["detected_issues"]:
                combined_analyses += "\nDetected issues:\n"
                for issue in analysis["detected_issues"]:
                    combined_analyses += f"  Line {issue['line']}: {issue['description']}\n"

            combined_analyses += "\n\n"

        # Check if the combined text is too large and truncate if necessary
        combined_tokens = len(self.model_manager.tokenizer.encode(combined_analyses))
        if combined_tokens > self.max_chunk_tokens * 0.9:
            print("Warning: Combined analyses exceed token limits, truncating...")
            return self._truncate_analyses(analyses)

        return combined_analyses

    def _truncate_analyses(self, analyses):
        """Truncate analyses to fit within token limits"""
        max_chars_per_analysis = int(30000 / len(analyses))
        truncated_analyses = ""

        for analysis in analyses:
            summary = analysis["analysis"]
            if len(summary) > max_chars_per_analysis:
                summary = summary[:max_chars_per_analysis] + "... [truncated]"

            truncated_analyses += (
                f"\n--- ANALYSIS OF CHUNK {analysis['chunk_number']}/{analysis['total_chunks']} ---\n"
                f"{summary}\n\n"
            )

        return truncated_analyses

    def _collect_all_issues(self, analyses):
        """Collect all detected issues from all analyses"""
        all_issues = []
        for analysis in analyses:
            all_issues.extend(analysis["detected_issues"])
        return all_issues

    def analyze_log_file(self, file_path, output_file=None):
        """Full pipeline to analyze a log file"""
        self._ensure_model_loaded()

        chunks = self.chunker.read_file_into_chunks(file_path)

        analyses = []
        for i, chunk in enumerate(chunks):
            analysis = self.analyze_chunk(chunk)
            analyses.append(analysis)

            if output_file:
                self._save_intermediate_result(analysis, output_file, i)

        final_result = self.synthesize_analyses(analyses)
        final_result["file_analyzed"] = file_path
        final_result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        final_result["model_used"] = self.model_manager.model_name

        if output_file:
            self._save_results(final_result, output_file)

        return final_result

    def _save_intermediate_result(self, analysis, output_file, chunk_index):
        """Save intermediate analysis result"""
        intermediate_file = f"{output_file}.chunk{chunk_index + 1}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved intermediate analysis to {intermediate_file}")

    def _save_results(self, final_result, output_file):
        """Save final analysis results to files"""
        # Save JSON result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2)
        print(f"\nSaved final analysis to {output_file}")

        # Save text version
        txt_output = f"{output_file}.txt"
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write("Log Analysis Report\n")
            f.write(f"File: {final_result['file_analyzed']}\n")
            f.write(f"Date: {final_result['timestamp']}\n")
            f.write(f"Model: {final_result['model_used']}\n")
            f.write(f"Chunks Analyzed: {final_result['chunk_count']}\n\n")
            f.write(final_result["final_analysis"])

            if final_result["detected_issues_combined"]:
                f.write("\n\n--- DETECTED ISSUES (with line numbers) ---\n")
                for issue in final_result["detected_issues_combined"]:
                    f.write(f"Line {issue['line']}: {issue['description']}\n")

        print(f"Saved text version to {txt_output}")

    def cleanup(self):
        """Clean up resources"""
        self.model_manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Analyze large log files with AI")
    parser.add_argument('--log_file', help='Path to the log file to analyze', default="./Jenkins_bad.txt")
    parser.add_argument('--output', '-o', help='Output file for the analysis (JSON format)')
    parser.add_argument('--model', default=Models.DEFAULT, help='Model to use for analysis')
    parser.add_argument('--cache-dir', default=None, help='Directory to cache the model')
    parser.add_argument('--max-chunk-tokens', type=int, default=10000, help='Maximum tokens per chunk')
    parser.add_argument('--overlap-tokens', type=int, default=200, help='Overlap tokens between chunks')

    args = parser.parse_args()

    # Set default cache directory if not specified
    if not args.cache_dir:
        args.cache_dir = Models.get_cache_dir(args.model)

    # Set default output file if not specified
    if not args.output:
        base_name = os.path.basename(args.log_file)
        args.output = f"analysis_{base_name}.json"

    # Create and configure analyzer
    analyzer = LogAnalyzer(model_name=args.model, cache_dir=args.cache_dir)
    analyzer.max_chunk_tokens = args.max_chunk_tokens
    analyzer.overlap_tokens = args.overlap_tokens

    try:
        # Perform analysis
        result = analyzer.analyze_log_file(args.log_file, args.output)

        # Print summary
        total_gen_time = sum(a['processing_time'] for a in result.get('analyses', []))
        print("\n=== Analysis Summary ===")
        print(f"File: {args.log_file}")
        print(f"Chunks analyzed: {result['chunk_count']}")
        print(f"Output: {args.output}")
        print(f"Model: {args.model}")
        print(f"Generation time (chunks): {total_gen_time:.2f}s, synthesis time: {result['processing_time']:.2f}s")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()
