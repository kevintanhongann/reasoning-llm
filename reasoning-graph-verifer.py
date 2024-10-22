import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
import networkx as nx
from tqdm import tqdm

class ReasoningGraphVerifier:
    def __init__(self, model_name: str = "gpt2-large", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        
    def create_reasoning_graph(self, prompt: str) -> nx.DiGraph:
        """Creates a directed graph representing the reasoning steps."""
        print("Generating reasoning steps...")
        reasoning_steps = self._generate_reasoning_steps(prompt)
        
        print("Creating graph...")
        graph = nx.DiGraph()
        
        for i, step in enumerate(tqdm(reasoning_steps, desc="Adding nodes and edges")):
            graph.add_node(i, content=step)
            if i > 0:
                graph.add_edge(i-1, i)
                
        return graph
    
    def _generate_reasoning_steps(self, prompt: str) -> List[str]:
        """Generates a list of reasoning steps using the LLM."""
        enhanced_prompt = f"Let's solve this step by step:\nQuestion: {prompt}\nSteps:"
        
        print("Generating response...")
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print("Decoding response...")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        steps = [step.strip() for step in response.split('\n') if step.strip()]
        return steps
    
    def verify_reasoning(self, graph: nx.DiGraph) -> Tuple[bool, List[str]]:
        """Verifies the reasoning graph for logical consistency."""
        verification_results = []
        is_valid = True
        
        print("Verifying reasoning...")
        for node in tqdm(graph.nodes(), desc="Checking nodes"):
            predecessors = list(graph.predecessors(node))
            if predecessors:
                current_step = graph.nodes[node]['content']
                prev_steps = [graph.nodes[pred]['content'] for pred in predecessors]
                
                consistency_check = self._verify_step_consistency(prev_steps, current_step)
                if not consistency_check[0]:
                    is_valid = False
                    verification_results.append(f"Inconsistency at step {node}: {consistency_check[1]}")
        
        return is_valid, verification_results
    
    def _verify_step_consistency(self, prev_steps: List[str], current_step: str) -> Tuple[bool, str]:
        """Verifies if a reasoning step is consistent with previous steps."""
        verification_prompt = "Given the previous steps:\n"
        for i, step in enumerate(prev_steps):
            verification_prompt += f"{i+1}. {step}\n"
        verification_prompt += f"\nIs the following step logically consistent: '{current_step}'?\nExplain why or why not."
        
        inputs = self.tokenizer(verification_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=256,
                temperature=0.3,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        verification = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        is_consistent = "inconsistent" not in verification.lower()
        return is_consistent, verification

class EnhancedReasoner:
    def __init__(self):
        self.verifier = ReasoningGraphVerifier()
        
    def solve_problem(self, prompt: str) -> Dict[str, Any]:
        """Solves a problem using graph-based reasoning verification."""
        print("Creating initial reasoning graph...")
        graph = self.verifier.create_reasoning_graph(prompt)
        
        print("Verifying reasoning...")
        is_valid, verification_results = self.verifier.verify_reasoning(graph)
        
        if not is_valid:
            print("Reasoning is invalid. Attempting to repair...")
            graph = self._repair_reasoning(graph, verification_results)
            print("Re-verifying repaired reasoning...")
            is_valid, verification_results = self.verifier.verify_reasoning(graph)
        
        print("Extracting final answer...")
        final_steps = [graph.nodes[node]['content'] for node in nx.topological_sort(graph)]
        
        return {
            'is_valid': is_valid,
            'reasoning_steps': final_steps,
            'verification_results': verification_results,
            'graph': graph
        }
    
    def _repair_reasoning(self, graph: nx.DiGraph, verification_results: List[str]) -> nx.DiGraph:
        """Attempts to repair invalid reasoning steps."""
        for result in tqdm(verification_results, desc="Repairing steps"):
            step_num = int(result.split('step')[1].split(':')[0])
            
            new_step = self.verifier._generate_reasoning_steps(
                f"Given the previous step: {graph.nodes[step_num-1]['content']}, what should the next step be?"
            )[0]
            
            graph.nodes[step_num]['content'] = new_step
            
        return graph

if __name__ == "__main__":
    reasoner = EnhancedReasoner()
    problem = "A farmer is traveling with a fox, a chicken, and a bag of grain. He must cross a river, but his boat can only carry himself and one other item. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How can he safely transport all three across the river?"
    print(f"Solving problem: {problem}")
    solution = reasoner.solve_problem(problem)
    print(f"Problem: {problem}")
    print(f"Is valid: {solution['is_valid']}")
    print("Reasoning steps:")
    for step in solution['reasoning_steps']:
        print(f"- {step}")
    print("Verification results:")
    if solution['verification_results']:
        for result in solution['verification_results']:
            print(f"- {result}")
    else:
        print("No verification issues found.")