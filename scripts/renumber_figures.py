"""Renumber figures in thesis.md: shift N → N+1 for figures 8-48.
This adds room for a new Figure 8 (constellation diagrams)."""
import re
import sys

filepath = 'thesis.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Process from highest to lowest to avoid double-renumbering
for old_num in range(48, 7, -1):
    new_num = old_num + 1
    # Match "Figure N:" and "Figure N " and "Figure N)" and "Figure N," etc.
    # In image alt text: ![Figure N:
    content = content.replace(f'Figure {old_num}:', f'Figure {new_num}:')
    content = content.replace(f'Figure {old_num} ', f'Figure {new_num} ')
    content = content.replace(f'Figure {old_num})', f'Figure {new_num})')
    content = content.replace(f'Figure {old_num},', f'Figure {new_num},')
    content = content.replace(f'Figure {old_num}.', f'Figure {new_num}.')
    
    # In list of figures table: "| N |" → "| N+1 |"  
    # Be careful with table entries — match "| 8 |" etc
    content = content.replace(f'| {old_num} |', f'| {new_num} |')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Done — renumbered Figures 8-48 → 9-49')
