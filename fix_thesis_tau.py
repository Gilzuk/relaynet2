with open('thesis_tau.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Find the appendix include line
idx = text.find('ch09_appendices')
with open('thesis_tau_context.txt', 'w', encoding='utf-8') as f:
    f.write(text[max(0,idx-200):idx+200])