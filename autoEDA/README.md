# autoEDA
Allows the user to produce a variety of charts by passing in only a datafame and a target column

From there, a handful of 1 line functions produce extensive output for further analysis:
*   plot_categorical()- Barplots w/ overlayed lines of repsonse % for all (or the top n) categortical features
*   plot_numeric()- Histograms w/ desity curves for all (or the top n) numeric features
*   plot_corr_heatmap()- Heatmap of the correlation between numeric features 
*   plot_scatterplots()- Scatterplots for the top n combinations of numeric features
*   plot_categorical_heatmaps()- Heatmaps for the top n combinations of numeric features
*   plot_numeric_categorical_pairs()- Violin/box plots for the top n combinations of numeric features and categortical features
*   plot_pca()- Pricipal Components Analysis of the numeric features with plotted variance exlained

See example_notebook.ipynb for usage...
Note: The fuctions have passable arguments, but they are not yet documented. Most functions have a 'max_plots' argument.

### Future Enhancements:
- regression
- muticlass
- smart toggle log of numericals (maybe if sd > mean * n)
- optional exclude list
- optional list of features to plot exclusivly
- allow lists to be passed to log_transform attributes
- smart annotate heat and corr maps depening on size
- warn about max plot issue 
- get around max plot issue (exclusion list is workaround)
