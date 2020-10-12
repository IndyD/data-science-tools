import itertools
import math
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.style as style
import scipy.stats as stats
import seaborn as sns
import warnings

from functools import wraps
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

import warnings


warnings.filterwarnings('ignore')
style.use('bmh') ## style for charts





class ClassificationEDA:
    """
    Allows the user to produce a variety of charts by passing in only a datafame and a target column

    From there, a handful of 1 line functions produce extensive output for further analysis:
    *   plot_categorical()- Barplots w/ overlayed lines of repsonse % for all (or the top n) categortical columns
    *   plot_numeric()- Histograms w/ desity curves for all (or the top n) numeric columns
    *   plot_corr_heatmap()- Heatmap of the correlation between numeric columns 
    *   plot_scatterplots()- Scatterplots for the top n combinations of numeric columns
    *   plot_categorical_heatmaps()- Heatmaps for the top n combinations of numeric columns
    *   plot_numeric_categorical_pairs()- Violin/box plots for top n combinations of numeric and categortical cols
    *   plot_pca()- Pricipal Components Analysis of the numeric columns with plotted variance exlained
    """

    def __init__(
        self, 
        df, 
        target,
        max_categories=None, 
    ):
        if df[target].nunique() != 2: raise ValueError('Target must have 2 unique values')
        if max_categories and max_categories < 2: raise ValueError('Max categories must be greater than 1')
        
        DEFAULT_MAX_CATEGORIES = 20
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object','bool','category']).columns
        numeric_cols = numeric_cols[numeric_cols != target]
        categorical_cols = categorical_cols[categorical_cols != target]
        
        self.target = target
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.max_categories = max_categories if max_categories is not None else DEFAULT_MAX_CATEGORIES
        self.df = self._clean_df(df)
        self._ranked_numeric_cols = self._rank_numeric_cols()
        self._ranked_categorical_cols = self._rank_categorical_cols()
        self._bar_lineplot_reference = None
                
    def _clean_df(self, df):
        df[self.target] = pd.get_dummies(df[self.target], drop_first=True)
        
        all_cols = np.concatenate([self.numeric_cols,self.categorical_cols,[self.target]])
        df = df[all_cols] #remove any non-numeric, non-categorical fields (ie dates)
        for col in self.categorical_cols:
            df[col].fillna("Unknown", inplace = True)
            top_categories = df[col].value_counts().nlargest(self.max_categories-1).index
            df.loc[~df[col].isin(top_categories), col] = "Other(Overflow)"
        
        return df
    
    def _rank_numeric_cols(self):
        correlations = [(col, abs(self.df[self.target].corr(self.df[col]))) for col in self.numeric_cols]
        correlations.sort(key=lambda tup: tup[1], reverse=True)
        ranked_numeric_cols = [col_corr[0] for col_corr in correlations]
        
        return ranked_numeric_cols
    
    def _rank_categorical_cols(self):
        ### Run a small batch of logistic regression with each categorical col to find significance
        sample_df = self.df.copy()
        if self.df.shape[0] > 1000: sample_df = self.df.sample(n=1000)

        col_scores = []
        for col in self.categorical_cols:
            y = sample_df[self.target]
            X_onehot = pd.get_dummies(sample_df[col], drop_first=True)

            lr = LogisticRegression(n_jobs=-1, max_iter=999)
            lr.fit(X_onehot,y)
            y_pred = lr.predict(X_onehot)
            acc = accuracy_score(y_pred, y)

            col_scores.append((col, acc))

        col_scores.sort(key=lambda tup: tup[1], reverse=True)
        ranked_categorical_cols = [col_corr[0] for col_corr in col_scores]
        
        return ranked_categorical_cols
    

    def _validate_min_numeric_cols(self, cols, min_cols):
        numeric_count = len(set(cols).intersection(self.numeric_cols))
        if numeric_count < min_cols:
            raise ValueError("Need at least {n} numeric columns".format(n=min_cols))

    def _validate_min_categorical_cols(self, cols, min_cols):
        categorical_count = len(set(cols).intersection(self.categorical_cols))
        if categorical_count < min_cols:
            raise ValueError("Need at least {n} categorical columns".format(n=min_cols)) 



    def _get_best_numeric_cols(self, cols, max_plots=40):
        self._validate_min_numeric_cols(cols, min_cols=1)
        ranked_plot_cols = [col for col in self._ranked_numeric_cols if col in cols]
        max_plots = max_plots if max_plots < len(ranked_plot_cols) else len(ranked_plot_cols)

        return ranked_plot_cols[0:max_plots]


    def _get_best_categorical_cols(self, cols, max_plots):
        self._validate_min_categorical_cols(cols, min_cols=1)
        ranked_plot_cols = [col for col in self._ranked_categorical_cols if col in cols]
        max_plots = max_plots if max_plots < len(ranked_plot_cols) else len(ranked_plot_cols)

        return ranked_plot_cols[0:max_plots]


    def _get_best_col_pairs(self, col_type, max_plots=40):
        # find how many cols are needed to satisfy the max scatter plot parameter
        n=2; m=1;
        while m < max_plots:
            m += n
            n += 1 

        if col_type == 'numeric':
            ranked_cols = self._ranked_numeric_cols 
        elif col_type == 'categorical':
            ranked_cols == self._ranked_categorical_cols 
        else:
            raise ValueError('Invalid col type')

        # if there aren't too many numerical proceed with using them all
        if len(ranked_cols) < n: n = len(ranked_cols)

        plot_cols = ranked_cols[0:n]
        weakest_col = plot_cols[n-1]

        # get all possible pairs 
        col_pairs = list(itertools.combinations(plot_cols, 2))

        while len(col_pairs) > max_plots:
            i = 0
            for col_pair in col_pairs:
                if col_pair[0] == weakest_col or col_pair[1] == weakest_col:
                    break
                i += 1
            col_pairs.pop(i)

        return col_pairs  

    def _get_best_numeric_pairs(self, max_plots):
        return self._get_best_col_pairs('numeric', max_plots)  

    def _get_categorical_best_pairs(self, max_plots):
        return self._get_best_col_pairs('categorical', max_plots)   
    
    def _get_best_numeric_categorical_pairs(self, max_plots=40):
        ### Find best pairs of numeric and categorical cols based on correlation and logistic regression score
        num_categoricals = math.ceil(math.sqrt(max_plots))
        if num_categoricals > len(self.categorical_cols): num_categoricals = len(self.categorical_cols) 
        
        num_numeric = math.ceil(max_plots/num_categoricals)
        if num_numeric > len(self.numeric_cols): num_numeric = len(self.numeric_cols) 

        categorical_pair_cols = self._ranked_categorical_cols[0:num_categoricals]
        numeric_pair_cols = self._ranked_numeric_cols[0:num_numeric]        
        weakest_numeric_col = numeric_pair_cols[num_numeric-1]

        categorical_numeric_pairs = [pair for pair in itertools.product(numeric_pair_cols, categorical_pair_cols)]
        ## if over max_plots limit, pop off pairs with the worst numerical col one at a time
        while len(categorical_numeric_pairs) > max_plots:
            i = 0
            for col_pair in categorical_numeric_pairs:
                if col_pair[0] == weakest_numeric_col or col_pair[1] == weakest_numeric_col:
                    break
                i += 1
            categorical_numeric_pairs.pop(i)

        return categorical_numeric_pairs
            

    
    
    def _log_transform_df(self, df, log_transform):
        # log log_transform is either True or a list af cols to log_transform
        log_transform = self.numeric_cols if log_transform is True else log_transform

        logged_cols = []
        for col in df:
            if col in log_transform and col in self.numeric_cols and min(df[col]) >= 0:
                # only positive values can be logged
                df[col] = np.log10(df[col] + 1)
                logged_cols.append(col)

        return df, logged_cols

    def _filter_cols_to_plot(self, possible_cols, specified_cols, exclude, filter_function, max_plots):
        if specified_cols: 
            possible_cols = specified_cols
        if exclude: 
            possible_cols = [col for col in possible_cols if col not in exclude]

        cols_to_plot = filter_function(possible_cols, max_plots)
        return cols_to_plot

    
    def plot_overlaid_barchart(self, field, verbose=False):
        ### Charts the counts of caterogical cols with % of binary response overlaid
        field_count  = self.df[field].value_counts()
        field_count_df = field_count.to_frame()
        field_count_df.columns = ['count']

        # Get the % target by category for the line overlay
        field_target_pct = pd.crosstab(self.df[field], self.df[self.target], normalize='index') * 100
        field_target_pct = field_target_pct.reset_index()
        # Try to choose the axis with smaller values to avoid skewed axis
        if not self._bar_lineplot_reference:
            self._bar_lineplot_reference = 1 if field_target_pct[0].median() < field_target_pct[1].median() else 2
        drop_index = self._bar_lineplot_reference
        field_target_pct = field_target_pct.drop(field_target_pct.columns[-drop_index],axis=1)

        merged_filed_target_pct = field_target_pct.merge(field_count_df, right_index=True, left_on=field)
        field_target_data = merged_filed_target_pct.sort_values('count', ascending=False).reset_index(drop=True)
        field_target_data.sort_values('count', ascending=False).reset_index(drop=True)
        
        if verbose: print(field_target_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel(field)
        ax = sns.barplot(
            field_target_data[field], 
            field_target_data['count'], 
            alpha=0.8
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_ylabel('count (bars)')
        ax2 = ax.twinx()
        ax2 = sns.pointplot(
            x=field_target_data[field], 
            y=field_target_data.iloc[:,-2], 
            color='black', 
            legend=False
        )
        ax2.set_ylabel('% {t} (line)'.format(t = self.target))
        plt.show()
        
    def plot_categorical(self, cols=False, exclude=None, max_plots=150, verbose=False):
        plot_cols = self._filter_cols_to_plot(
            possible_cols = self.categorical_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_categorical_cols, 
            max_plots = max_plots
        )
        self._validate_min_categorical_cols(plot_cols, min_cols=1)
        
        plt.figure(figsize=(10, 6*max_plots))
        for col in plot_cols:
            self.plot_overlaid_barchart(col, verbose)
            
            
    def plot_numeric(self, cols=False, exclude=None, max_plots=150, bins=25, log_transform=False):
        plot_cols = self._filter_cols_to_plot(
            possible_cols = self.numeric_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_numeric_cols, 
            max_plots = max_plots
        )
        self._validate_min_numeric_cols(plot_cols, min_cols=1)

        plot_df = self.df.copy()
        # ensure single index dosen't become Series
        plot_df = pd.DataFrame(plot_df[ plot_cols + [self.target] ])
        if log_transform: 
            plot_df, logged_cols = self._log_transform_df(plot_df, log_transform)

        target_value0 = plot_df[self.target].value_counts().index[0]
        target_value1 = plot_df[self.target].value_counts().index[1]

        fig, axes = plt.subplots(len(plot_cols),figsize=(10, 6*len(plot_cols)))
        for i, col in enumerate(plot_cols):
            col_name = 'log_{c}'.format(c=col) if col in logged_cols else col
            
            sns.distplot(
                plot_df.loc[plot_df[self.target] == target_value0][col],
                label=target_value0, 
                bins = bins, 
                ax=axes[i]
            )
            sns.distplot(
                plot_df.loc[plot_df[self.target] != target_value0][col],
                label=target_value1, 
                bins = bins, 
                ax=axes[i]
            )
            axes[i].legend(loc='upper right')
            axes[i].set_title('{c} histogram'.format(c=col_name))
    
    
    def plot_scatterplots(self, cols=False, alpha=0.6, log_transform=False, max_plots=40):
        if len(self.numeric_cols) < 2: raise ValueError('Need at least 2 numeric cols for scatterplots')
        if cols and not set(cols).issubset(set(self.numeric_cols)): raise ValueError("Invalid 'cols' argument")

        scatter_pairs = self._get_numeric_best_pairs(max_plots=max_plots)
        f, axes = plt.subplots(len(scatter_pairs),figsize=(10, 6*len(scatter_pairs)))
        
        for i, pair in enumerate(scatter_pairs):
            plot_df = self.df.copy()[[pair[0], pair[1], self.target]]
            
            logf0 = ''
            logf1 = ''
            
            if log_transform:
                if min(plot_df[pair[0]]) >= 0:
                    plot_df[pair[0]] = np.log10(plot_df[pair[0]] + 1)
                    logf0 = 'log_'
                if min(plot_df[pair[1]]) >= 0:
                    plot_df[pair[1]] = np.log10(plot_df[pair[1]] + 1)
                    logf1 = 'log_'
            title = '{p0}{f0} vs {p1}{f1}'.format(p0=logf0, f0=pair[0], p1=logf1, f1=pair[1])
            
            sns.scatterplot(
                data=plot_df, 
                x=pair[0], 
                y=pair[1], 
                hue=self.df[self.target].tolist(), 
                alpha=alpha, 
                ax=axes[i]
            )
            axes[i].set_title(title)
            

    def plot_numeric_categorical_pairs(self, max_plots=40, boxplot_only=False, log_transform=False):
        if len(self.categorical_cols) == 0: raise ValueError('Need at least 1 categorical col')
        if len(self.numeric_cols) == 0: raise ValueError('Need at least 1 numeric cols')
            
        plot_df = self.df.copy()
        num_cat_pairs = self._get_best_numeric_categorical_pairs(max_plots)
        
        fig, axes = plt.subplots(len(num_cat_pairs),figsize=(10, 6*len(num_cat_pairs)))
        for i, pair in enumerate(num_cat_pairs):
            numeric_col_name = pair[0]
            if log_transform:
                if min(plot_df[numeric_col_name]) >= 0:
                    plot_df[numeric_col_name] = np.log10(plot_df[numeric_col_name] + 1)
                    numeric_col_name = 'log_{f}'.format(f=numeric_col_name)

            title = '{c} vs {n}'.format(c=pair[1], n=numeric_col_name)
            category_count = len(self.df[pair[1]].value_counts())

            if category_count <= 15 and not boxplot_only:
                sns.violinplot(
                    x=pair[1], 
                    y=pair[0], 
                    hue=self.target, 
                    data=plot_df, 
                    split=True, 
                    inner='quart', 
                    ax=axes[i]
                )
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
            else:
                sns.boxplot(x=pair[1], y=pair[0], hue=self.target, data=plot_df, ax=axes[i])

            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
            axes[i].set_title(title)
                

    def plot_categorical_heatmaps(self, max_plots=50):
        if len(self.categorical_cols) < 2: raise ValueError('Need at least 2 categorical cols for heatmap')
        categorical_pairs = self._get_categorical_best_pairs(max_plots)
        
        f, axes = plt.subplots(len(categorical_pairs),figsize=(10, 6*len(categorical_pairs)))
        for i, pair in enumerate(categorical_pairs):
            sns.heatmap(pd.pivot_table(self.df,index=[pair[0]], values=self.target, columns=[pair[1]]), ax=axes[i])
            
    def plot_pca(self, output_components=None):
        ## Perform PCA and plot variability described the PCs
        if len(self.numeric_cols) < 2: raise ValueError('Need at least 2 numeric cols for PCA')
        pca_df = self.df[self.numeric_cols].copy()
            
        imp=SimpleImputer(missing_values=np.NaN)
        imp_df=pd.DataFrame(imp.fit_transform(pca_df))
            
        pca = PCA(n_components=imp_df.shape[1])
        pca.fit(imp_df)

        ## Output error explained by sqrt(n)th term
        if not output_components: output_components = math.floor(math.sqrt(imp_df.shape[1]))

        ## Inspect the explained variances to determine how many components to use  
        plt.subplots(figsize=(8, 8))
        # use n_components series to make x axis start at 1
        n_components = pd.Series(range(1,len(np.cumsum(pca.explained_variance_ratio_))+1))
        plt.plot(n_components, np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')

        ## Output the explained variances at output_components # of components
        output_str = 'Cumulative Explained variance at {n} PCA components:'.format(n=output_components)
        print(output_str,sum(pca.explained_variance_ratio_[0:output_components]) )
    
    def plot_corr_heatmap(self, annot=False):
        if len(self.numeric_cols) < 2: raise ValueError('Need at least 2 numeric cols for corr heatmap')
        df_numeric = self.df[self.numeric_cols]
        sns.heatmap(df_numeric.corr(), annot=annot)