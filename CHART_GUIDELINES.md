# Chart Guidelines

1. **Unique visual identity**: Every curve must use a unique combination of color AND marker — no repeats.
2. **Thin lines**: Use thin lines (linewidth ≤ 1.5), never bold. Add an inset zoom area for congested regions.
3. **Legend placement**: Place the legend outside the plot area or in empty space — never overlapping curves.
4. **Annotation leaders**: Add leader lines with text annotations in congested regions where curves are hard to distinguish.
5. **Curve jitter**: When two curves share the same BER value, apply a small multiplicative offset (e.g., ×1.03 / ×0.97) so they remain visually separable.
6. **Y-axis focus**: Set the y-axis lower bound one decade below the minimum nonzero BER in the data (e.g., if min BER ≈ 1e-2, set ylim down to 1e-3). Do not show many empty decades.
8. **Baseline curves**: Plot AF and DF baselines in grey/black with thin lines, clearly labeled in the legend.
7. **Variant line styles**: Use distinct line styles (solid, dashed, dash-dot, custom dashes) to differentiate activation variants (e.g., with/without LN).
9. **Consistent color palette**: Use a consistent 30-color palette across all charts, ensuring no color is reused for different curves.
10. **Plot dimensions**: Use a consistent aspect ratio (e.g., 1.5:1 width:height) and ensure all plots are large enough to clearly show details (e.g., 8×5 inches).
11. **Font sizes**: Use readable font sizes for titles, axis labels, legends, and annotations (e.g., title: 16pt, axes: 14pt, legend: 12pt).
12. **Gridlines**: Add light gridlines to improve readability, but ensure they do not overpower the data.
13. **Colorblind-friendly**: Ensure the color palette is colorblind-friendly, using tools like ColorBrewer to select appropriate colors.
14. **Consistent markers**: Use a consistent set of markers (e.g., circle, square, triangle, diamond) across all charts to represent different models or variants.
15. **Zoomed insets**: For regions where curves are close together, add zoomed inset plots to show details without cluttering the main plot.
16. **Annotation of key points**: Annotate key points (e.g., where one activation outperforms another) with arrows and text to highlight important insights.
17. **Consistent file naming**: Save plots with descriptive and consistent file names that reflect the content (e.g., `activation_comparison_qpsk.png`).
18. **BER summary tables**: In addition to plots, provide BER summary tables at key SNR points (e.g., 4 dB and 16 dB) with Δ BER between activations for quick reference.
19. provide summary charts that aggregate results across constellations and models, using bar charts or heatmaps to show overall trends in activation performance.
20. provide achievement chart where the winner of each constellation/model is highlighted, showing which activation performed best in each scenario while the others are faded out.
21. **JSON persistence**: Every experiment run must save full BER results (mean, per-trial, 95% CI bounds) to a `.json` file alongside the plots, enabling later chart regeneration without re-running the experiment.
22. **Top-3 chart**: After each experiment, automatically generate a focused chart showing only the 3 best-performing neural architectures compared to AF and DF baselines. Ranking is by average BER across the upper half of the SNR range. Use bold distinct colors for the top-3 and grey/black for baselines.

