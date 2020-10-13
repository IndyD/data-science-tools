import itertools
import logging
import math
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import matplotlib.style as style
import scipy.stats as stats
import seaborn as sns
import warnings

from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
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
        DEFAULT_MAX_CATEGORIES = 20
        self._validate_input(df, target, max_categories)
        
        max_categories = max_categories if max_categories is not None else DEFAULT_MAX_CATEGORIES
        
        numeric_cols = df.select_dtypes(include=['int8','int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object','bool','category']).columns
        numeric_cols = set(numeric_cols[numeric_cols != target])
        categorical_cols = set(categorical_cols[categorical_cols != target])
        combined_cols = numeric_cols.union(categorical_cols)

        all_cols = combined_cols.union([target])
        unusable_cols = list(set(df.columns) - set(all_cols))
        if len(unusable_cols) > 0:
            log.warning('Unable to use the following colunms: {uc}'.format(uc=unusable_cols))

        df = df[all_cols] #remove any non-numeric, non-categorical fields (ie dates)
        df[target] = pd.get_dummies(df[target], drop_first=True) # converts the target to 1 or 0
        df = self._df_bin_max_categories(df, categorical_cols, max_categories)
        
        self.target = target
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.combined_cols = combined_cols
        self.all_cols = all_cols
        self.max_categories = max_categories
        
        self.df = df
        self._ranked_numeric_cols = self._rank_numeric_cols()
        self._ranked_categorical_cols = self._rank_categorical_cols()
        self._bar_lineplot_reference = None

    def _validate_input(self, df, target, max_categories):
        """ Validate the input of class instantiation """
        if not isinstance(target, str): raise ValueError('Invalid target: {t}'.format(t=target))
        if not isinstance(df, pd.DataFrame): raise ValueError('Invalid input, please input a pandas DataFrame')
        if target not in df.columns: raise ValueError('Target not in dataframe: {t}'.format(t=target))
        if df.shape[1] < 2: raise ValueError('Dataframe must have at least 2 columns')
        # 10 is an arbitrary limit
        if df.shape[0] < 10: raise ValueError('Dataframe must have at least 10 rows')
        if df[target].nunique() != 2: raise ValueError('Target must have 2 unique values')
        if max_categories and max_categories < 2: raise ValueError('Max categories must be greater than 1')


    def _df_bin_max_categories(self, df, categorical_cols, max_categories):
        """ Cap the max number of categories in categorical fields for readability """
        for col in categorical_cols:
            df[col].fillna("Unknown", inplace = True)
            top_categories = df[col].value_counts().nlargest(max_categories-1).index
            # set values with ranked counts below max_categories to "Other(Overflow)"
            df.loc[~df[col].isin(top_categories), col] = "Other(Overflow)"
        
        return df
    
    def _rank_numeric_cols(self):
        """ Rank the numeric functions by correlation to the target """
        correlations = [(col, abs(self.df[self.target].corr(self.df[col]))) for col in self.numeric_cols]
        correlations.sort(key=lambda tup: tup[1], reverse=True)
        ranked_numeric_cols = [col_corr[0] for col_corr in correlations]
        
        return ranked_numeric_cols
    
    def _rank_categorical_cols(self):
        """ Run small batch of logistic regression against target with each categorical col to rank """
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

        # rank based on training set accuracy
        col_scores.sort(key=lambda tup: tup[1], reverse=True)
        ranked_categorical_cols = [col_corr[0] for col_corr in col_scores]
        
        return ranked_categorical_cols
    
    def _is_listlike(self, parameter):
        return isinstance(parameter, (list, tuple, set, pd.Series, pd.Index))

    def _validate_min_numeric_cols(self, cols, min_cols):
        """ Validate that at least n colunms are numeric in cols list (n=min_cols)"""
        if cols is False: 
            numeric_count = len(self.numeric_cols)
        else:
            numeric_count = len(set(cols).intersection(self.numeric_cols))
        if numeric_count < min_cols:
            raise ValueError("Need at least {n} numeric columns".format(n=min_cols))

    def _validate_min_categorical_cols(self, cols, min_cols):
        """ Validate that at least n colunms are categorical in cols list (n=min_cols)"""
        if cols is False: 
            categorical_count = len(self.categorical_cols)
        else:
            categorical_count = len(set(cols).intersection(self.categorical_cols))
        if categorical_count < min_cols:
            raise ValueError("Need at least {n} categorical columns".format(n=min_cols)) 

    def _get_best_numeric_cols(self, cols, max_plots):
        """ Find top n ranked numeric columns in cols list (n=max_plots)"""
        self._validate_min_numeric_cols(cols, min_cols=1)
        ranked_plot_cols = [col for col in self._ranked_numeric_cols if col in cols]
        max_plots = max_plots if max_plots < len(ranked_plot_cols) else len(ranked_plot_cols)
        return ranked_plot_cols[0:max_plots]

    def _get_best_categorical_cols(self, cols, max_plots):
        """ Find top n ranked categorical columns in cols list (n=max_plots)"""
        self._validate_min_categorical_cols(cols, min_cols=1)
        ranked_plot_cols = [col for col in self._ranked_categorical_cols if col in cols]
        max_plots = max_plots if max_plots < len(ranked_plot_cols) else len(ranked_plot_cols)

        return ranked_plot_cols[0:max_plots]

    def _get_best_col_pairs(self, ranked_cols, max_plots):
        """ Find top n pairs of columns in ranked_cols list (n=max_plots)"""
        # n is how many columns are needed to satisfy the pairs criteria
        n=2; m=1;
        while m < max_plots:
            m += n
            n += 1 

        # if the number of columns is less than n, use them all
        if len(ranked_cols) <= n: n = len(ranked_cols)
        plot_cols = ranked_cols[0:n]
        weakest_col = plot_cols[n-1]

        # get all possible pairs 
        col_pairs = list(itertools.combinations(plot_cols, 2))
        # remove the excess using the weakest column (the nth column)
        while len(col_pairs) > max_plots:
            i = 0
            for col_pair in col_pairs:
                if col_pair[0] == weakest_col or col_pair[1] == weakest_col:
                    break
                i += 1
            col_pairs.pop(i)

        return col_pairs  

    def _get_best_numeric_pairs(self, cols, max_plots):
        """ Find top n pairs of ranked numeric columns in cols list (n=max_plots)"""
        self._validate_min_numeric_cols(cols, min_cols=2)
        ranked_cols = [col for col in self._ranked_numeric_cols if col in cols]
        return self._get_best_col_pairs(ranked_cols, max_plots)  

    def _get_best_categorical_pairs(self, cols, max_plots):
        """ Find top n pairs of ranked categorical columns in cols list (n=max_plots)"""
        self._validate_min_categorical_cols(cols, min_cols=2)
        ranked_cols = [col for col in self._ranked_categorical_cols if col in cols]
        return self._get_best_col_pairs(ranked_cols, max_plots)   
    
    def _get_best_numeric_categorical_pairs(self, cols, max_plots):
        """ Find top n ranked pairs of (numeric, categorical) columns in cols list (n=max_plots)"""
        self._validate_min_categorical_cols(cols, min_cols=1)
        self._validate_min_numeric_cols(cols, min_cols=1)
        ranked_categorical_cols = [col for col in self._ranked_categorical_cols if col in cols]
        ranked_numeric_cols = [col for col in self._ranked_numeric_cols if col in cols]

        ## Find best pairs of numeric and categorical cols based on correlation and logistic regression score
        # try to get an even split, preferring categorical
        num_categoricals = math.ceil(math.sqrt(max_plots))
        if num_categoricals > len(ranked_categorical_cols): num_categoricals = len(ranked_categorical_cols) 
        num_numeric = math.ceil(max_plots/num_categoricals)
        if num_numeric > len(ranked_numeric_cols): num_numeric = len(ranked_numeric_cols) 

        categorical_pair_cols = ranked_categorical_cols[0:num_categoricals]
        numeric_pair_cols = ranked_numeric_cols[0:num_numeric]

        weakest_numeric_col = numeric_pair_cols[num_numeric-1]

        categorical_numeric_pairs = [pair for pair in itertools.product(numeric_pair_cols, categorical_pair_cols)]
        # if over max_plots limit, pop off pairs with the worst numerical col one at a time
        while len(categorical_numeric_pairs) > max_plots:
            i = 0
            for col_pair in categorical_numeric_pairs:
                if col_pair[0] == weakest_numeric_col or col_pair[1] == weakest_numeric_col:
                    break
                i += 1
            categorical_numeric_pairs.pop(i)

        return categorical_numeric_pairs
        
    def _log_transform_df(self, df, log_transform):
        """ Take log base 10 of the specified columns in the log_transform parameter """
        logged_cols = []

        # log_transform can be: True, a string, or an iterable of cols to transform
        if log_transform is True:
            transform_cols = self.numeric_cols.intersection(set(df.columns))
        elif isinstance(log_transform, str):
            transform_cols = [log_transform]
        elif self._is_listlike(log_transform):
            transform_cols = log_transform
        else: raise ValueError('Invalid argument to log_tranform parameter: {l}'.format(l=log_transform))

        for col in transform_cols:
            if col not in self.numeric_cols: 
                log.warning("Unable to log transform non-numeric column: {c}".format(c=col))

        for col in df:
            # only positive values can be logged
            if col in transform_cols and col in self.numeric_cols and min(df[col]) >= 0:
                df[col] = np.log10(df[col] + 1)
                logged_cols.append(col)
            elif col in transform_cols and col in self.numeric_cols and min(df[col]) < 0:
                log.warning("Unable to log transform column with negative values: {c}".format(c=col))

        return df, logged_cols

    def _create_transformed_plot_df(self, plot_cols, log_transform):
        """ Create local copy of df for plot and log transform """
        plot_df = self.df.copy()
        # wrap in DataFrame() to ensure single index doesn't become Series
        plot_df = pd.DataFrame(plot_df[ list(plot_cols) + [self.target] ])

        if log_transform: 
            plot_df, logged_cols = self._log_transform_df(plot_df, log_transform)
        else: logged_cols = []

        return plot_df, logged_cols


    def _filter_cols_to_plot(self, possible_cols, specified_cols, exclude, filter_function, max_plots): 
        """ Apply parameters specified by the user to find list of column/bivariates to plot """
        if not isinstance(max_plots, int): raise ValueError('Max_plots must be an integer')

        if specified_cols: 
            if isinstance(specified_cols, str):
                specified_cols = [specified_cols]
            if not self._is_listlike(specified_cols): 
                raise ValueError('Invalid cols argument: {c}'.format(c=specified_cols))
            invalid_col = list(set(specified_cols) - self.all_cols)
            if len(invalid_col) > 0:
                log.error('Invalid colums passed to cols parameter: {i}'.format(i=invalid_col))
            possible_cols = specified_cols

        if exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            if not self._is_listlike(exclude): 
                raise ValueError('Invalid cols argument: {c}'.format(c=exclude))
            invalid_exclude = list(set(exclude) - self.combined_cols)
            if len(invalid_exclude) > 0:
                log.error('Invalid colums passed to exclude parameter: {i}'.format(i=invalid_exclude))
            if self.target in exclude: log.warning("Can't exclude target column")
            possible_cols = [col for col in possible_cols if col not in exclude]

        cols_to_plot = filter_function(possible_cols, max_plots)
        return cols_to_plot

    
    def plot_overlaid_barchart(self, field, verbose=False):
        """ Charts the counts of caterogical cols with % of binary response overlaid """ 
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
        if verbose: print(field_target_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel(field)
        ax = sns.barplot(
            field_target_data[field], 
            field_target_data['count'], 
            alpha=0.8,
            order = field_target_data.sort_values('count', ascending=False)[field]
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_ylabel('count (bars)')
        ax2 = ax.twinx() # dual axis graph
        # line graph of % target in category
        ax2 = sns.pointplot(
            x=field_target_data[field], 
            y=field_target_data.iloc[:,-2], 
            color='black', 
            legend=False
        )
        ax2.set_ylabel('% {t} (line)'.format(t = self.target))
        plt.show()
        
    def plot_categorical(self, cols=False, exclude=None, max_plots=150, verbose=False):
        """ Take all categorical and plot bar charts with overlayed line charts of target pct """
        plot_cols = self._filter_cols_to_plot(
            possible_cols = self.categorical_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_categorical_cols, 
            max_plots = max_plots
        )
        self._validate_min_categorical_cols(plot_cols, min_cols=1)
        
        for col in plot_cols:
            self.plot_overlaid_barchart(col, verbose)
            
    
    def plot_numeric(self, cols=False, exclude=None, max_plots=150, bins=25, log_transform=False):
        """ Take all numeric and overlayed histograms with density curves for the two values of the target """
        plot_cols = self._filter_cols_to_plot(
            possible_cols = self.numeric_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_numeric_cols, 
            max_plots = max_plots
        )
        self._validate_min_numeric_cols(plot_cols, min_cols=1)
        plot_df, logged_cols = self._create_transformed_plot_df(plot_cols, log_transform)

        target_value0 = plot_df[self.target].value_counts().index[0]
        target_value1 = plot_df[self.target].value_counts().index[1]

        for col in plot_cols:
            # prefix 'log_' to the colunm name if it was log transformed
            col_name = 'log_{c}'.format(c=col) if col in logged_cols else col
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.distplot(
                plot_df.loc[plot_df[self.target] == target_value0][col],
                label=target_value0, 
                bins = bins, 
            )
            sns.distplot(
                plot_df.loc[plot_df[self.target] != target_value0][col],
                label=target_value1, 
                bins = bins, 
            )
            ax.legend(loc='upper right')
            ax.set_title('{c} histogram'.format(c=col_name))
    
    def plot_scatterplots(self, cols=False, exclude=None, alpha=0.6, log_transform=False, max_plots=40):
        """ Plot satterplots of pairs of numeric columns colored by target """
        scatter_pairs = self._filter_cols_to_plot(
            possible_cols = self.numeric_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_numeric_pairs, 
            max_plots = max_plots
        )
        plot_cols = set([col for pair in scatter_pairs for col in pair])
        self._validate_min_numeric_cols(plot_cols, min_cols=2)
        plot_df, logged_cols = self._create_transformed_plot_df(plot_cols, log_transform)
        
        for pair in scatter_pairs:
            fig, ax = plt.subplots(figsize=(10, 6))
            prefix0 = 'log_' if pair[0] in logged_cols else ''
            prefix1 = 'log_' if pair[1] in logged_cols else ''
            title = '{p0}{f0} vs {p1}{f1}'.format(p0=prefix0, f0=pair[0], p1=prefix1, f1=pair[1])
            
            sns.scatterplot(
                data=plot_df, 
                x=pair[0], 
                y=pair[1], 
                hue=plot_df[self.target].tolist(), 
                alpha=alpha, 
            )
            ax.set_title(title)


    def plot_categorical_heatmaps(self, cols=False, exclude=None, max_plots=50, annot=False):
        """ Plot pairs categorical columns against the target """
        categorical_pairs = self._filter_cols_to_plot(
            possible_cols = self.categorical_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_categorical_pairs, 
            max_plots = max_plots
        )
        plot_cols = set([col for pair in categorical_pairs for col in pair])
        self._validate_min_categorical_cols(plot_cols, min_cols=2)
        
        for pair in categorical_pairs:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pd.pivot_table(self.df,index=[pair[0]], values=self.target, columns=[pair[1]]),
                annot=annot,
            ) 


    def plot_numeric_categorical_pairs(
            self, 
            cols=False, 
            exclude=None, 
            boxplot_only=False, 
            log_transform=False,
            max_plots=40
        ):
        """ Plot pairs of numeric vs categorical columns broken down by the target """
        self._validate_min_categorical_cols(cols, min_cols=1)
        self._validate_min_numeric_cols(cols, min_cols=1)
            
        num_cat_pairs = self._filter_cols_to_plot(
            possible_cols = self.combined_cols, 
            specified_cols = cols, 
            exclude = exclude, 
            filter_function = self._get_best_numeric_categorical_pairs, 
            max_plots = max_plots
        )
        plot_cols = set([col for pair in num_cat_pairs for col in pair])
        plot_df, logged_cols = self._create_transformed_plot_df(plot_cols, log_transform)

        
        for pair in num_cat_pairs:
            fig, ax = plt.subplots(figsize=(10, 6))
            # prefix 'log_' to the colunm name if it was log transformed
            numeric_col_name = 'log_{c}'.format(c=pair[0]) if pair[0] in logged_cols else pair[0]
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
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            else:
                sns.boxplot(x=pair[1], y=pair[0], hue=self.target, data=plot_df)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_title(title)
                

    def plot_pca(self, output_components=None):
        """ Perform PCA and plot variability described the PCs """
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
        """ Plot grid of numeric columns with a heat map of their correlations """
        if len(self.numeric_cols) < 2: raise ValueError('Need at least 2 numeric cols for corr heatmap')
        df_numeric = self.df[self.numeric_cols]
        sns.heatmap(df_numeric.corr(), annot=annot)