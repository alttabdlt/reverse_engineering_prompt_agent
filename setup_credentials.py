#!/usr/bin/env python3
"""
Setup script to help configure Google Cloud credentials for the project.
"""

import os
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

def main():
    console.print("[bold cyan]🔐 Google Cloud Credentials Setup[/bold cyan]")
    console.print("="*50)
    console.print("\nThis script will help you configure credentials for Vertex AI.\n")
    
    # Check current setup
    current_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    current_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    
    if current_path:
        console.print(f"[yellow]Current GOOGLE_APPLICATION_CREDENTIALS:[/yellow] {current_path}")
        if os.path.exists(current_path):
            console.print("[green]✓ File exists[/green]")
        else:
            console.print("[red]✗ File not found[/red]")
    
    if current_json:
        console.print("[yellow]GOOGLE_APPLICATION_CREDENTIALS_JSON is set[/yellow] (JSON content)")
    
    # Ask user how they want to configure
    console.print("\n[bold]Choose configuration method:[/bold]")
    console.print("1. Provide path to existing service account JSON file")
    console.print("2. Paste service account JSON content")
    console.print("3. Use existing Google Cloud CLI authentication")
    
    choice = Prompt.ask("Select option", choices=["1", "2", "3"])
    
    env_file = Path(".env")
    env_content = []
    
    # Read existing .env if it exists
    if env_file.exists():
        with open(env_file) as f:
            env_content = [line.strip() for line in f.readlines() 
                         if line.strip() and not line.startswith("GOOGLE_APPLICATION_CREDENTIALS")]
    
    if choice == "1":
        # Path to file
        while True:
            file_path = Prompt.ask("\n[green]Enter path to service account JSON file[/green]")
            file_path = os.path.expanduser(file_path)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path) as f:
                        json.load(f)  # Validate it's valid JSON
                    
                    console.print(f"[green]✓ Valid service account file found[/green]")
                    
                    # Add to .env
                    env_content.append(f"GOOGLE_APPLICATION_CREDENTIALS={os.path.abspath(file_path)}")
                    break
                    
                except json.JSONDecodeError:
                    console.print("[red]Error: File is not valid JSON[/red]")
                except Exception as e:
                    console.print(f"[red]Error reading file: {e}[/red]")
            else:
                console.print(f"[red]File not found: {file_path}[/red]")
                if not Confirm.ask("Try again?"):
                    return
    
    elif choice == "2":
        # JSON content
        console.print("\n[green]Paste your service account JSON content:[/green]")
        console.print("[dim](Paste the entire JSON, then press Enter twice)[/dim]\n")
        
        json_lines = []
        blank_count = 0
        
        while True:
            line = input()
            if not line:
                blank_count += 1
                if blank_count >= 2:
                    break
            else:
                blank_count = 0
                json_lines.append(line)
        
        json_content = '\n'.join(json_lines)
        
        try:
            # Validate JSON
            parsed = json.loads(json_content)
            if "type" in parsed and parsed["type"] == "service_account":
                console.print("[green]✓ Valid service account JSON[/green]")
                
                # Escape for shell
                escaped_json = json_content.replace("'", "'\"'\"'")
                env_content.append(f"GOOGLE_APPLICATION_CREDENTIALS_JSON='{escaped_json}'")
            else:
                console.print("[red]Error: Not a valid service account JSON[/red]")
                return
                
        except json.JSONDecodeError:
            console.print("[red]Error: Invalid JSON format[/red]")
            return
    
    elif choice == "3":
        # Use default credentials
        console.print("\n[yellow]Using Application Default Credentials[/yellow]")
        console.print("Make sure you've run: [cyan]gcloud auth application-default login[/cyan]")
        # Don't add any credential env vars
    
    # Check for project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        project_id = Prompt.ask("\n[green]Enter your Google Cloud Project ID[/green]")
        env_content.append(f"GOOGLE_CLOUD_PROJECT={project_id}")
    
    # Check for location
    location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    console.print(f"\n[yellow]Current Vertex AI location:[/yellow] {location}")
    if Confirm.ask("Change location?"):
        console.print("[dim]Common locations: us-central1, us-east1, europe-west4, asia-northeast1[/dim]")
        location = Prompt.ask("Enter location")
        env_content.append(f"VERTEX_AI_LOCATION={location}")
    else:
        env_content.append(f"VERTEX_AI_LOCATION={location}")
    
    # Write .env file
    if env_content:
        with open(env_file, 'w') as f:
            f.write('\n'.join(env_content) + '\n')
        console.print(f"\n[green]✓ Configuration saved to .env[/green]")
    
    # Test the configuration
    console.print("\n[bold]Testing configuration...[/bold]")
    
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        import vertexai
        from vertexai.preview.generative_models import GenerativeModel
        
        # Re-read env vars after loading .env
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        
        if choice == "1":
            from google.oauth2 import service_account
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            credentials = service_account.Credentials.from_service_account_file(creds_path)
            vertexai.init(project=project_id, location=location, credentials=credentials)
        elif choice == "2":
            from google.oauth2 import service_account
            import tempfile
            creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            creds_data = json.loads(creds_json)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(creds_data, f)
                temp_path = f.name
            credentials = service_account.Credentials.from_service_account_file(temp_path)
            vertexai.init(project=project_id, location=location, credentials=credentials)
            os.unlink(temp_path)
        else:
            vertexai.init(project=project_id, location=location)
        
        # Try to initialize model
        model = GenerativeModel('gemini-1.5-flash')
        console.print("[green]✓ Successfully connected to Vertex AI![/green]")
        
        # Offer to test
        if Confirm.ask("\nTest with a simple prompt?"):
            response = model.generate_content("Say 'Hello from Gemini!'")
            console.print(f"\n[cyan]Gemini says:[/cyan] {response.text}")
        
    except Exception as e:
        console.print(f"[red]Error testing configuration: {e}[/red]")
        console.print("\n[yellow]Please check your credentials and try again.[/yellow]")
        return
    
    console.print("\n[bold green]✅ Setup complete![/bold green]")
    console.print("\nYou can now run:")
    console.print("[cyan]python interactive_demo.py[/cyan]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled.[/yellow]")