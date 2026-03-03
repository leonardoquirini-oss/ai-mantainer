#!/usr/bin/env python3
"""
Script di avvio per l'agente di gestione manutenzioni
"""

import sys
import os

# Aggiungi la directory corrente al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.main import main

if __name__ == "__main__":
    main()
