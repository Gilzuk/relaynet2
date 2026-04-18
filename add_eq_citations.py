"""
Add missing \cite{} source annotations to equations in ch01 and ch02.
Maps equation labels to their canonical citation keys.
"""
import re

# Mapping: equation label -> citation key(s)
eq_citations = {
    # ch01 - Information Theory / Relay Capacity
    'eq:relay-capacity-bound': 'cover1979capacity',
    'eq:df-capacity': 'cover1979capacity',
    'eq:df-gaussian-capacity': 'cover1979capacity',
    'eq:af-gain': 'laneman2004cooperative',
    'eq:af-snr': 'laneman2004cooperative',
    'eq:df-ber-hops': 'proakis2008digital',
    # ch01 - MIMO
    'eq:mimo-capacity': 'telatar1999capacity',
    'eq:system-model': 'tse2005fundamentals',
    'eq:zf-equalizer': 'tse2005fundamentals',
    'eq:zf-snr': 'tse2005fundamentals',
    'eq:mmse-equalizer': 'tse2005fundamentals',
    'eq:mmse-sinr': 'tse2005fundamentals',
    'eq:sic-step': 'tse2005fundamentals',
    # ch01 - Channel Models
    'eq:awgn-channel': 'proakis2008digital',
    'eq:awgn-ber': 'proakis2008digital',
    'eq:rayleigh-ber': 'simon2005digital',
    'eq:rayleigh-ber-approx': 'simon2005digital',
    'eq:rician-pdf': 'simon2005digital',
    'eq:rician-ber': 'simon2005digital',
    'eq:fading-coefficient': 'tse2005fundamentals',
    # ch01 - SSM / Mamba
    'eq:ssm-continuous': 'gu2022efficiently',
    'eq:ssm-discrete': 'gu2022efficiently',
    'eq:mamba-selective': 'gu2024mamba',
    'eq:ssd-kernel': 'dao2024transformers',
    # ch01 - Attention
    'eq:attention': 'vaswani2017attention',
    'eq:multihead-attention': 'vaswani2017attention',
    'eq:positional-encoding': 'vaswani2017attention',
    # ch01 - GAN
    'eq:gan-minimax': 'goodfellow2014generative',
    'eq:wasserstein-distance': 'gulrajani2017improved',
    'eq:gradient-penalty': 'gulrajani2017improved',
    # ch01 - VAE
    'eq:vae-elbo': 'kingma2014auto',
    # ch02 - NN Theory
    'eq:optimal-denoiser': 'goodfellow2016deep',
    'eq:optimal-denoiser-awgn': 'goodfellow2016deep',
    'eq:bias-variance-decomp': 'goodfellow2016deep',
    'eq:mse-loss': 'goodfellow2016deep',
    # ch04 - Methods
    'eq:ber-estimate': 'proakis2008digital',
    'eq:wilcoxon-test': 'proakis2008digital',
    'eq:qpsk-mapping': 'proakis2008digital',
    'eq:qam16-mapping': 'proakis2008digital',
    'eq:psk16-mapping': 'proakis2008digital',
    'eq:qam16-ber': 'proakis2008digital',
    'eq:psk16-ber': 'proakis2008digital',
    'eq:power-normalization': 'proakis2008digital',
    'eq:iq-splitting': 'proakis2008digital',
    'eq:joint-classification': 'goodfellow2016deep',
    'eq:confidence-interval': 'proakis2008digital',
    'eq:relay-received-siso': 'laneman2004cooperative',
    'eq:zf-equalizer-methods': 'tse2005fundamentals',
    'eq:mmse-equalizer-methods': 'tse2005fundamentals',
}

def add_citation_to_eq(content, label, cite_key):
    """Add \par{\small\textit{Source: \cite{key}}} after equation with given label."""
    # Find the equation block containing this label
    pattern = r'(\\begin\{equation\}.*?\\label\{' + re.escape(label) + r'\}.*?)(\\end\{equation\})'
    
    def replacer(m):
        block = m.group(1)
        end = m.group(2)
        # Only add if not already cited
        if '\\cite' not in block and '\\par{\\small' not in block:
            return block + end + r'\par{\small\textit{Source: \cite{' + cite_key + r'}}}'
        return m.group(0)
    
    return re.sub(pattern, replacer, content, flags=re.DOTALL)

# Process ch01 and ch02
for filename in ['chapters/ch01_introduction.tex', 'chapters/ch02_literature_review.tex']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for label, cite_key in eq_citations.items():
        content = add_citation_to_eq(content, label, cite_key)
    
    if content != original:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Updated {filename}')

print('Done adding equation citations.')