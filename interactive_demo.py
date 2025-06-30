#!/usr/bin/env python3
"""
Interactive Reverse Prompt Engineering Demo

This script allows users to:
1. Enter a prompt
2. Generate output from that prompt
3. See if the detective can reverse engineer the original prompt
"""

import asyncio
import aiohttp
import json
import os
import sys
from typing import Dict, Any
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Override system env vars with .env file

console = Console()

# Import Vertex AI
try:
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    console.print("[red]Error: Vertex AI not installed. Run: pip install google-cloud-aiplatform[/red]")
    sys.exit(1)

class PromptGenerator:
    """Generates output from user prompts using Gemini via Vertex AI"""
    
    def __init__(self):
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        
        if not project_id:
            console.print("[red]Error: GOOGLE_CLOUD_PROJECT not set in environment[/red]")
            sys.exit(1)
        
        try:
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            
            if creds_path:
                if os.path.exists(creds_path):
                    console.print(f"[dim]Using credentials from: {creds_path}[/dim]")
                    credentials = service_account.Credentials.from_service_account_file(creds_path)
                    vertexai.init(project=project_id, location=location, credentials=credentials)
                else:
                    console.print(f"[red]Error: Credentials file not found at: {creds_path}[/red]")
                    console.print("\n[yellow]To fix this, do one of the following:[/yellow]")
                    console.print("1. Place your service account JSON at that location")
                    console.print("2. Update GOOGLE_APPLICATION_CREDENTIALS to point to your actual key file:")
                    console.print("   [cyan]export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json[/cyan]")
                    console.print("3. Or set the JSON content directly:")
                    console.print("   [cyan]export GOOGLE_APPLICATION_CREDENTIALS_JSON='$(cat /path/to/your/key.json)'[/cyan]")
                    sys.exit(1)
            elif creds_json:
                # Parse JSON string
                import tempfile
                try:
                    creds_data = json.loads(creds_json)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(creds_data, f)
                        temp_path = f.name
                    credentials = service_account.Credentials.from_service_account_file(temp_path)
                    vertexai.init(project=project_id, location=location, credentials=credentials)
                    os.unlink(temp_path)
                    console.print("[dim]Using credentials from GOOGLE_APPLICATION_CREDENTIALS_JSON[/dim]")
                except json.JSONDecodeError:
                    console.print("[red]Error: GOOGLE_APPLICATION_CREDENTIALS_JSON contains invalid JSON[/red]")
                    sys.exit(1)
            else:
                # Try default credentials
                console.print("[dim]No explicit credentials found, trying application default credentials...[/dim]")
                vertexai.init(project=project_id, location=location)
            
            # Initialize model
            self.model = GenerativeModel('gemini-1.5-flash')
            console.print("[green]✓ Gemini initialized successfully via Vertex AI[/green]")
            
        except Exception as e:
            console.print(f"[red]Error initializing Vertex AI: {str(e)}[/red]")
            sys.exit(1)
    
    def generate_output(self, prompt: str) -> str:
        """Generate output from the given prompt"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                )
            )
            return response.text.strip()
        except Exception as e:
            console.print(f"[red]Error generating output: {str(e)}[/red]")
            return None

async def analyze_output(output_text: str) -> Dict[str, Any]:
    """Send output to the detective API for reverse engineering"""
    url = "http://localhost:8000/analyze"
    
    async with aiohttp.ClientSession() as session:
        try:
            payload = {
                "output_text": output_text,
                "max_attempts": 5,
                "context": "Interactive reverse engineering demo"
            }
            
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {"success": False, "error": f"API Error {response.status}: {error_text}"}
                
                return await response.json()
                
        except Exception as e:
            return {"success": False, "error": str(e)}

def display_results(original_prompt: str, generated_output: str, detection_result: Dict[str, Any]):
    """Display the results in a nice format"""
    console.print("\n" + "="*80)
    console.print("[bold cyan]🔍 REVERSE PROMPT ENGINEERING RESULTS[/bold cyan]")
    console.print("="*80 + "\n")
    
    # Original prompt
    console.print(Panel(
        original_prompt,
        title="[bold green]1️⃣ Your Original Prompt[/bold green]",
        border_style="green"
    ))
    
    # Generated output
    console.print(Panel(
        generated_output[:500] + ("..." if len(generated_output) > 500 else ""),
        title="[bold blue]2️⃣ Generated Output (sent to detective)[/bold blue]",
        border_style="blue"
    ))
    
    if detection_result.get("success"):
        result = detection_result["result"]
        hypothesis = result["best_hypothesis"]
        
        # Detected prompt
        console.print(Panel(
            hypothesis["prompt"],
            title="[bold yellow]3️⃣ Detective's Reconstructed Prompt[/bold yellow]",
            border_style="yellow"
        ))
        
        # Confidence and reasoning
        confidence_color = "green" if hypothesis["confidence"] > 0.8 else "yellow" if hypothesis["confidence"] > 0.6 else "red"
        console.print(f"\n[bold]Confidence:[/bold] [{confidence_color}]{hypothesis['confidence']:.1%}[/{confidence_color}]")
        
        console.print(f"\n[bold]Reasoning:[/bold] {hypothesis['reasoning']}")
        
        # Key elements
        if hypothesis.get("key_elements"):
            console.print("\n[bold]Key Elements Identified:[/bold]")
            for elem in hypothesis["key_elements"]:
                console.print(f"  • {elem}")
        
        # Enhanced scoring if available
        if result.get("enhanced_scoring"):
            scoring = result["enhanced_scoring"]
            console.print("\n[bold]Detailed Analysis:[/bold]")
            console.print(f"  • Semantic similarity: {scoring.get('semantic_similarity', 0):.1%}")
            console.print(f"  • Structural match: {scoring.get('structural_match', 0):.1%}")
            console.print(f"  • Style match: {scoring.get('style_match', 0):.1%}")
        
        # Alternative hypotheses
        if result.get("alternative_hypotheses"):
            console.print("\n[bold]Alternative Hypotheses:[/bold]")
            for i, alt in enumerate(result["alternative_hypotheses"][:2], 1):
                console.print(f"  {i}. {alt['prompt']} (confidence: {alt['confidence']:.1%})")
        
        # Success evaluation
        console.print("\n[bold]Evaluation:[/bold]")
        
        # Simple similarity check
        original_lower = original_prompt.lower()
        detected_lower = hypothesis["prompt"].lower()
        
        # Check for key words match
        original_words = set(original_lower.split())
        detected_words = set(detected_lower.split())
        word_overlap = len(original_words.intersection(detected_words)) / len(original_words)
        
        if word_overlap > 0.7 or hypothesis["confidence"] > 0.85:
            console.print("[bold green]✅ EXCELLENT MATCH![/bold green] The detective successfully reverse-engineered your prompt!")
        elif word_overlap > 0.5 or hypothesis["confidence"] > 0.7:
            console.print("[bold yellow]🔶 GOOD MATCH![/bold yellow] The detective captured the essence of your prompt.")
        else:
            console.print("[bold red]❌ PARTIAL MATCH[/bold red] The detective found a related but different prompt.")
            
    else:
        console.print(Panel(
            f"[red]Error: {detection_result.get('error', 'Unknown error')}[/red]",
            title="[bold red]❌ Detection Failed[/bold red]",
            border_style="red"
        ))

async def check_server():
    """Check if the API server is running"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/health") as response:
                return response.status == 200
        except:
            return False

async def main():
    """Main interactive loop"""
    console.print("[bold cyan]🔮 Interactive Reverse Prompt Engineering Demo[/bold cyan]")
    console.print("="*60)
    console.print("This demo will:")
    console.print("1. Take your prompt")
    console.print("2. Generate output using Gemini")
    console.print("3. Send ONLY the output to the detective")
    console.print("4. See if it can reverse engineer your original prompt!")
    console.print("="*60 + "\n")
    
    # Check server
    if not await check_server():
        console.print("[red]❌ API server not running![/red]")
        console.print("Please start it with: [yellow]cd part1 && python start_server.py[/yellow]")
        return
    
    # Initialize generator
    generator = PromptGenerator()
    
    while True:
        # Get user prompt
        console.print("\n[bold]Enter a prompt to test reverse engineering:[/bold]")
        console.print("[dim](Examples: 'Write a haiku about coding', 'List 5 benefits of exercise', 'Explain quantum computing simply')[/dim]")
        
        user_prompt = Prompt.ask("\n[green]Your prompt[/green]")
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            console.print("\n[yellow]Thanks for testing! 👋[/yellow]")
            break
        
        # Generate output
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Generate output
            task1 = progress.add_task("[cyan]Generating output from your prompt...", total=None)
            generated_output = generator.generate_output(user_prompt)
            progress.update(task1, completed=100)
            
            if not generated_output:
                console.print("[red]Failed to generate output. Please try again.[/red]")
                continue
            
            # Analyze with detective
            task2 = progress.add_task("[yellow]Detective analyzing the output...", total=None)
            result = await analyze_output(generated_output)
            progress.update(task2, completed=100)
        
        # Display results
        display_results(user_prompt, generated_output, result)
        
        # Ask to continue
        console.print("\n" + "-"*60)
        if not Prompt.ask("\n[cyan]Try another prompt?[/cyan]", choices=["y", "n"], default="y") == "y":
            console.print("\n[yellow]Thanks for testing! 👋[/yellow]")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Goodbye! 👋[/yellow]")