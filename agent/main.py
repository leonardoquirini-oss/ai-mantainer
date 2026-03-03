#!/usr/bin/env python3
"""
CLI principale per l'agente di gestione manutenzioni
"""

import typer
from datetime import date, datetime
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="maintainer",
    help="Agente di gestione manutenzioni per mezzi di trasporto"
)
console = Console()


@app.command()
def carica(
    filepath: str = typer.Argument(..., help="Path al file CSV con storico manutenzioni")
):
    """Carica dati storici di manutenzione da CSV"""
    from agent.tools.maintenance_tools import carica_dati_csv

    console.print(f"[bold]Caricamento dati da {filepath}...[/bold]")

    try:
        result = carica_dati_csv(filepath)
        if result["success"]:
            console.print(f"[green]✓ Caricati {result['eventi_caricati']} eventi per {result['mezzi_caricati']} mezzi[/green]")
            console.print(f"  Tipi mezzo: {', '.join(result['tipi_mezzo'])}")
            console.print(f"  Tipi guasto: {', '.join(result['tipi_guasto'])}")
        else:
            console.print(f"[red]Errore nel caricamento[/red]")
    except FileNotFoundError:
        console.print(f"[red]File non trovato: {filepath}[/red]")
    except Exception as e:
        console.print(f"[red]Errore: {e}[/red]")


@app.command()
def mezzi(
    stato: Optional[str] = typer.Option(None, help="Filtra per stato (attivo/fermo/in_manutenzione)")
):
    """Lista mezzi disponibili"""
    console.print("[bold]Lista mezzi[/bold]")
    console.print("(Da implementare con connettore database)")


@app.command()
def manutenzioni(
    data: str = typer.Argument(None, help="Data nel formato YYYY-MM-DD"),
    tipo: Optional[str] = typer.Option(None, help="Tipo manutenzione")
):
    """Lista manutenzioni programmate"""
    if data:
        try:
            data_parsed = datetime.strptime(data, "%Y-%m-%d").date()
        except ValueError:
            console.print("[red]Formato data non valido. Usa YYYY-MM-DD[/red]")
            return
    else:
        data_parsed = date.today()

    console.print(f"[bold]Manutenzioni per {data_parsed}[/bold]")
    console.print("(Da implementare con connettore database)")


@app.command()
def predizioni(
    mesi: int = typer.Option(12, help="Orizzonte temporale in mesi")
):
    """Genera predizioni di manutenzione basate su analisi NHPP"""
    from agent.tools.maintenance_tools import get_previsioni_guasti

    console.print(f"[bold]Predizioni manutenzione (prossimi {mesi} mesi)[/bold]\n")

    previsioni = get_previsioni_guasti(mesi)

    if not previsioni:
        console.print("[yellow]Nessun dato disponibile. Carica prima i dati con 'carica <file.csv>'[/yellow]")
        return

    table = Table(title=f"Previsioni Guasti - {mesi} mesi")
    table.add_column("Mezzo", style="cyan")
    table.add_column("Tipo", style="blue")
    table.add_column("Guasti Attesi", justify="right", style="red")
    table.add_column("Prossimo Guasto (mesi)", justify="right", style="yellow")
    table.add_column("Trend", style="magenta")

    for p in previsioni[:20]:  # Mostra top 20
        table.add_row(
            p["mezzo_id"],
            p["tipo_mezzo"],
            f"{p['guasti_attesi']:.1f}",
            f"{p['tempo_prossimo_guasto']:.1f}" if p['tempo_prossimo_guasto'] else "-",
            p["trend"]
        )

    console.print(table)


@app.command()
def statistiche():
    """Mostra statistiche dataset manutenzioni"""
    from agent.tools.maintenance_tools import get_statistiche_dataset

    console.print("[bold]Statistiche Dataset Manutenzioni[/bold]\n")

    stats = get_statistiche_dataset()

    if stats["totale_eventi"] == 0:
        console.print("[yellow]Nessun dato disponibile. Carica prima i dati con 'carica <file.csv>'[/yellow]")
        return

    table = Table(title="Riepilogo Dataset")
    table.add_column("Metrica", style="cyan")
    table.add_column("Valore", style="green")

    table.add_row("Eventi totali", str(stats["totale_eventi"]))
    table.add_row("Mezzi totali", str(stats["totale_mezzi"]))
    table.add_row("Tipi mezzo", ", ".join(stats["tipi_mezzo"]))
    table.add_row("Tipi guasto", ", ".join(stats["tipi_guasto"]))

    if "eventi_per_tipo_mezzo" in stats:
        for tipo, count in stats["eventi_per_tipo_mezzo"].items():
            table.add_row(f"  - {tipo}", str(count))

    console.print(table)


@app.command()
def piano(
    affidabilita: float = typer.Option(0.90, help="Affidabilità target (0-1)"),
    output: Optional[str] = typer.Option(None, help="File output per il report")
):
    """Genera piano di manutenzione ottimizzato basato su analisi statistica"""
    from agent.tools.maintenance_tools import genera_piano_manutenzione

    console.print(f"[bold]Generazione piano manutenzione (affidabilità target: {affidabilita*100:.0f}%)[/bold]\n")

    result = genera_piano_manutenzione(affidabilita)

    if not result.get("intervalli_manutenzione"):
        console.print("[yellow]Nessun dato disponibile. Carica prima i dati con 'carica <file.csv>'[/yellow]")
        return

    # Mostra report testuale
    console.print(result["report"])

    # Tabella intervalli
    table = Table(title="Intervalli Manutenzione Consigliati")
    table.add_column("Tipo Mezzo", style="cyan")
    table.add_column("Tipo Guasto", style="blue")
    table.add_column("Intervallo (mesi)", justify="right", style="green")
    table.add_column("Applicabile", style="yellow")
    table.add_column("Motivazione", style="dim")

    for i in result["intervalli_manutenzione"]:
        table.add_row(
            i["tipo_mezzo"],
            i["tipo_guasto"],
            str(i["intervallo_mesi"]) if i["intervallo_mesi"] else "-",
            "✓" if i["applicabile"] else "✗",
            i["motivazione"][:50] + "..." if len(i["motivazione"]) > 50 else i["motivazione"]
        )

    console.print(table)

    # Mezzi critici
    if result["mezzi_critici"]:
        console.print(f"\n[red bold]⚠ Mezzi critici (deterioramento): {len(result['mezzi_critici'])}[/red bold]")
        for m in result["mezzi_critici"][:5]:
            console.print(f"  - {m}")

    if output:
        with open(output, "w") as f:
            f.write(result["report"])
        console.print(f"\n[green]Report salvato in: {output}[/green]")


@app.command()
def test_connessione():
    """Testa la connessione ai sistemi"""
    console.print("[bold]Test connessione sistemi...[/bold]")
    console.print("[yellow]Connessione database: (da implementare)[/yellow]")


@app.command()
def chat():
    """Avvia chat interattiva con l'agente LLM"""
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt

    console.print(Panel(
        "[bold blue]Agente Manutenzione Predittiva[/bold blue]\n\n"
        "Digita le tue richieste in linguaggio naturale.\n"
        "Comandi speciali:\n"
        "  [green]/quit[/green] - Esci\n"
        "  [green]/reset[/green] - Resetta conversazione\n"
        "  [green]/help[/green] - Mostra aiuto",
        title="Benvenuto"
    ))

    # Crea agente
    try:
        from agent.llm_agent import create_agent
        agent = create_agent()
    except ValueError as e:
        console.print(f"[red]Errore: {e}[/red]")
        console.print("[yellow]Imposta la variabile d'ambiente OPENROUTER_API_KEY[/yellow]")
        raise typer.Exit(1)

    console.print()

    while True:
        try:
            # Prompt utente
            user_input = Prompt.ask("[bold green]Tu[/bold green]")

            if not user_input.strip():
                continue

            # Comandi speciali
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().split()[0]

                if cmd in ["quit", "exit", "q", "esci"]:
                    console.print("[yellow]Arrivederci![/yellow]")
                    break

                elif cmd == "reset":
                    agent.reset_conversation()
                    console.print("[green]Conversazione resettata[/green]")
                    continue

                elif cmd == "help":
                    console.print(Panel(
                        "[green]/quit[/green] - Esci\n"
                        "[green]/reset[/green] - Resetta conversazione\n"
                        "[green]/help[/green] - Mostra questo aiuto\n\n"
                        "[bold]Esempi di domande:[/bold]\n"
                        "- Mostrami le statistiche del dataset\n"
                        "- Genera un piano di manutenzione\n"
                        "- Analizza i guasti ai freni per i trattori\n"
                        "- Quali sono i mezzi critici?\n"
                        "- Previsioni guasti per i prossimi 12 mesi",
                        title="Aiuto"
                    ))
                    continue

                else:
                    console.print(f"[yellow]Comando sconosciuto: /{cmd}[/yellow]")
                    continue

            # Messaggio normale - invia all'agente
            with console.status("[bold green]Elaborazione..."):
                response = agent.chat(user_input)

            console.print()
            console.print(Panel(Markdown(response), title="[bold blue]Agente[/bold blue]"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrotto. Usa /quit per uscire.[/yellow]")
            continue


def main():
    """Entry point principale"""
    app()


if __name__ == "__main__":
    main()
