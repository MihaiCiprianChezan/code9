import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import time
import os
import gc
import argparse
import math
import json
from optimum.bettertransformer import BetterTransformer


# Model names and default configuration
Q_0_5B = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
Q_1_5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
Q_3B = "Qwen/Qwen2.5-Coder-3B-Instruct"
OC = "open-r1/OlympicCoder-7B"

DEFAULT_MODEL = Q_1_5B
DEFAULT_MODEL_CACHE_DIR = "./model_cache__" + DEFAULT_MODEL.split("/")[-1].replace(".", "_")


class LogAnalyzer:
    def __init__(self, model_name, cache_dir=None):
        """Initialize the log analyzer with model configuration."""
        # Set environment variables for optimization
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))

        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

        # Adjustable chunk settings
        self.max_chunk_tokens = 30000  # Maximum tokens per chunk; adjust as needed
        self.overlap_tokens = 200  # Overlap between chunks for context
        self.max_analysis_tokens = 3000  # Max tokens for generated analysis

        # System prompt configurations for detailed analysis and synthesis
        self.chunk_system_prompt = (
            "You are a log analysis expert. Analyze this log file segment. "
            "Look for errors, critical issues, warnings, unusual patterns, and security concerns. "
            "Give only a summary of the most important findings. Avoid redundancy. "
            "Be concise but thorough. "
            "Focus on the most important findings in this segment and be short and brief in explanations. "
            "Prioritise and highlight by criticality. Discard less critical or less important findings."
        )

        self.summarization_system_prompt = (
            "You are a log analysis expert. I will provide you with analyses of different segments "
            "of a large log file. Synthesize these analyses into one unified and comprehensive summary, "
            "not as individual chunks or segments. "
            "Identify all patterns across all segments into one analysis, avoid redundancy, "
            "and focus on the most important issues. Categorize findings by severity (Critical, Error, Warning). Avoid Info type and less critical findings. "
            "Highlight any root causes and security concerns."
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def print_model_device_map(self):
        """Print where each model component is placed."""
        print("\nModel device map:")
        for name, module in self.model.named_modules():
            if hasattr(module, 'device'):
                print(f"{name}: {module.device}")
            elif hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                print(f"{name}: {module.weight.device}")

    def load_model(self):
        """Load the model with optimizations."""
        print(f"Loading model {self.model_name}...")

        # Free up memory before loading model
        gc.collect()
        torch.cuda.empty_cache()

        # Print CUDA availability information - add this
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")

        # Try to load from cached directory if available
        if self.cache_dir:
            model_path = os.path.join(self.cache_dir, "log_analyzer_model")
            if os.path.exists(model_path):
                print("Loading from cached model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                # Load model with all optimizations
                self.model = self._load_fresh_model()
                # Cache model for future use
                os.makedirs(self.cache_dir, exist_ok=True)
                self.model.save_pretrained(model_path)
        else:
            self.model = self._load_fresh_model()
            # self.model  = self.model.half()

        # Load model and move to GPU
        # self.model = self._load_fresh_model()
        # self.model = self.model.to(self.device)
        # self.model = self.model.eval()

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
        # self.print_model_device_map()

    def _load_fresh_model(self):
        try:
            # First attempt to load with Flash Attention 2
            print("Attempting to load model with Flash Attention...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation="flash_attention_2",  # Explicitly request Flash Attention 2
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
        return BetterTransformer.transform(model)

    def _warm_up_model(self):
        """Warm up the model to initialize CUDA kernels."""
        print("Warming up model...")
        dummy_input = "Hello, this is a warm-up."
        inputs = self.tokenizer([dummy_input], return_tensors="pt").to(self.model.device)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = self.model(**{k: v for k, v in inputs.items() if k != 'token_type_ids'}, max_length=1)

    def read_file_into_chunks(self, file_path):
        """
        Read the specified file line by line and split it into chunks based on token counts.
        Each chunk contains its starting line number so that issues can be referenced accurately.
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

        while line_index < total_lines:
            line = lines[line_index].rstrip("\n")
            token_count_line = len(self.tokenizer.encode(line, add_special_tokens=False))
            if current_token_count + token_count_line > self.max_chunk_tokens and chunk_lines:
                chunk_text = "\n".join(chunk_lines)
                chunks.append({
                    "text": chunk_text,
                    "start_line": start_line,
                    "chunk_number": len(chunks) + 1,
                    "total_chunks": None  # to be updated later
                })
                # Implement overlap: reuse some lines from the end
                overlap_line_count = 3
                overlap_lines = chunk_lines[-overlap_line_count:] if len(chunk_lines) >= overlap_line_count else chunk_lines
                chunk_lines = overlap_lines.copy()
                current_token_count = sum(len(self.tokenizer.encode(l, add_special_tokens=False)) for l in chunk_lines)
                start_line = line_index - len(chunk_lines) + 1
            else:
                chunk_lines.append(line)
                current_token_count += token_count_line
                line_index += 1

        if chunk_lines:
            chunk_text = "\n".join(chunk_lines)
            chunks.append({
                "text": chunk_text,
                "start_line": start_line,
                "chunk_number": len(chunks) + 1,
                "total_chunks": None
            })

        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        estimated_tokens = sum(len(self.tokenizer.encode(line, add_special_tokens=False)) for line in lines) * 1.3
        print(f"Estimated file size: ~{estimated_tokens:.0f} tokens")
        print(f"File split into {total_chunks} chunks.")

        return chunks

    def analyze_chunk(self, chunk):
        """
        Analyze a single chunk of the log file.
        In addition to model generation, scan the chunk text for error keywords and attach line numbers.
        """
        chunk_number = chunk["chunk_number"]
        total_chunks = chunk["total_chunks"]

        print(f"\nAnalyzing chunk {chunk_number}/{total_chunks} (starting at line {chunk['start_line']})")

        prompt = (
            f"This is chunk {chunk_number} of {total_chunks} from a log file. Analyze it for important issues:\n\n"
            f"{chunk['text']}"
        )

        messages = [
            {"role": "system", "content": self.chunk_system_prompt},
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        start_time = time.time()
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        with torch.amp.autocast('cuda', enabled=True):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # output = self.model.generate(
                #     **model_inputs,
                #     max_new_tokens=self.max_analysis_tokens,
                #     do_sample=False,
                #     use_cache=True,
                #     temperature=1.0,
                #     streamer=streamer,
                #     num_beams=1,
                #     pad_token_id=self.tokenizer.pad_token_id,
                # )
                output = self.model.generate(
                    **model_inputs,
                    do_sample=True,  # Enable sampling
                    top_p=0.9,
                    top_k=10,
                    streamer=streamer,
                    max_new_tokens=self.max_analysis_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=0.8,
                    num_beams=1,
                    use_cache=True,
                    early_stopping=True
                )

        end_time = time.time()

        output_text = self.tokenizer.decode(
            output[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Scan the chunk text for possible issues based on keywords (adding explicit line numbers)
        detected_issues = []
        for offset, line in enumerate(chunk["text"].split("\n")):
            if "ERROR" in line or "Exception" in line:
                detected_issues.append({
                    "line": chunk["start_line"] + offset,
                    "description": line.strip()
                })

        analysis = {
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "analysis": output_text,
            "processing_time": end_time - start_time,
            "detected_issues": detected_issues
        }

        return analysis

    def synthesize_analyses(self, analyses):
        """Synthesize individual chunk analyses into a comprehensive summary."""
        print("\nSynthesizing all analyses into a comprehensive report...")

        combined_analyses = ""
        all_detected_issues = []
        for analysis in analyses:
            combined_analyses += f"\n--- ANALYSIS OF CHUNK {analysis['chunk_number']}/{analysis['total_chunks']} ---\n"
            combined_analyses += analysis["analysis"]
            if analysis["detected_issues"]:
                combined_analyses += "\nDetected issues:\n"
                for issue in analysis["detected_issues"]:
                    combined_analyses += f"  Line {issue['line']}: {issue['description']}\n"
                    all_detected_issues.append(issue)
            combined_analyses += "\n\n"

        combined_tokens = self.tokenizer.encode(combined_analyses, add_special_tokens=False)
        if len(combined_tokens) > self.max_chunk_tokens * 0.9:
            print("Warning: Combined analyses exceed token limits, truncating...")
            temp = ""
            for analysis in analyses:
                summary = analysis["analysis"]
                max_chars = int(30000 / len(analyses))
                if len(summary) > max_chars:
                    summary = summary[:max_chars] + "... [truncated]"
                temp += f"\n--- ANALYSIS OF CHUNK {analysis['chunk_number']}/{analysis['total_chunks']} ---\n{summary}\n\n"
            combined_analyses = temp

        prompt = f"Here are analyses of different segments of a log file. Synthesize them into one comprehensive report:\n\n{combined_analyses}"

        messages = [
            {"role": "system", "content": self.summarization_system_prompt},
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        print("\n--- Generating comprehensive analysis ---\n")
        streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        start_time = time.time()
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                streamer=streamer,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        end_time = time.time()

        final_analysis = self.tokenizer.decode(
            output[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return {
            "final_analysis": final_analysis,
            "chunk_count": len(analyses),
            "processing_time": end_time - start_time,
            "analyses": analyses,
            "detected_issues_combined": all_detected_issues
        }

    def analyze_log_file(self, file_path, output_file=None):
        """Full pipeline to analyze a log file."""
        if not self.model:
            self.load_model()

        chunks = self.read_file_into_chunks(file_path)

        analyses = []
        for i, chunk in enumerate(chunks):
            analysis = self.analyze_chunk(chunk)
            analyses.append(analysis)

            if output_file:
                intermediate_file = f"{output_file}.chunk{i + 1}.json"
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
                print(f"Saved intermediate analysis to {intermediate_file}")

        final_result = self.synthesize_analyses(analyses)
        final_result["file_analyzed"] = file_path
        final_result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        final_result["model_used"] = self.model_name

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2)
            print(f"\nSaved final analysis to {output_file}")

            txt_output = f"{output_file}.txt"
            with open(txt_output, 'w', encoding='utf-8') as f:
                f.write("Log Analysis Report\n")
                f.write(f"File: {file_path}\n")
                f.write(f"Date: {final_result['timestamp']}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Chunks Analyzed: {final_result['chunk_count']}\n\n")
                f.write(final_result["final_analysis"])
                if final_result["detected_issues_combined"]:
                    f.write("\n\n--- DETECTED ISSUES (with line numbers) ---\n")
                    for issue in final_result["detected_issues_combined"]:
                        f.write(f"Line {issue['line']}: {issue['description']}\n")
            print(f"Saved text version to {txt_output}")

        return final_result

    def cleanup(self):
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Analyze large log files with AI")
    parser.add_argument('--log_file', help='Path to the log file to analyze', default="./Torch_Log.txt")
    parser.add_argument('--output', '-o', help='Output file for the analysis (JSON format)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Model to use for analysis')
    parser.add_argument('--cache-dir', default=DEFAULT_MODEL_CACHE_DIR, help='Directory to cache the model')
    parser.add_argument('--max-chunk-tokens', type=int, default=30000, help='Maximum tokens per chunk')
    parser.add_argument('--overlap-tokens', type=int, default=200, help='Overlap tokens between chunks')

    args = parser.parse_args()

    analyzer = LogAnalyzer(model_name=args.model, cache_dir=args.cache_dir)
    analyzer.max_chunk_tokens = args.max_chunk_tokens
    analyzer.overlap_tokens = args.overlap_tokens

    if not args.output:
        base_name = os.path.basename(args.log_file)
        args.output = f"analysis_{base_name}.json"

    try:
        result = analyzer.analyze_log_file(args.log_file, args.output)

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