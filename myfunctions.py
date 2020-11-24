import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm

#-----------------------------
# Import feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


#--------------------------------------------------------------------
#
#   Custom functions
#
#--------------------------------------------------------------------

def divider(phrase:str="New Problem"):
    print('\n#{:-^67}#\n# {}\n#{:-^67}#\n'.format('',phrase,''))
    return

def topmarker(phrase:str="", marker:str="-"):
    print('\n{:{}^70}\n'.format(phrase, marker))
    return

def border(marker:str = "~"):
    print('{:{}^70}\n'.format("",marker))
    return


#--------------------------------------------------------------------
def viewmultiv(data, name="", annot=False, pairplot=True, steps=7):

    if name != "":
        name = name.rstrip()
        name = name

    if pairplot:
        # pairplot takes a lot of time to generate
        # therefore option to skip
        # scatterplot matrix   
        fig1 = sns.pairplot(data)
        fig1.savefig('Pairplot '+name+'.png', dpi=300)
    
    # correlation  aka heatmap matrix

    sns.set(style="white")
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    fig2, ax  = plt.subplots(figsize=(10, 7))

    if steps < 7: steps = 7
    cmap = sns.color_palette("BrBG", steps)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
            square=True, annot=annot, linewidths=.5, cbar_kws={"shrink": .5})
     
    fig2.tight_layout()
    fig2.savefig('CorrelationMatrix '+name+'.png', dpi=300)

    return

#--------------------------------------------------------------------
def viewhistograms(data, kde=True):
    # takes a dataframe and creates histograms of each column

    try:
        names = data.columns.values
    except AttributeError:
        names = data.name

    n = len(data)
    k2 = int(np.ceil(2 * ( n ** (2/5))))   # variable bin width method

    i = 0

    for col in data.columns:
        title = 'Histogram of ' + names[i]
        # fig_hist = plt.figure()
        fig_hist = plt.figure(figsize=(8, 4))

        sns.histplot(data[col], kde=kde, bins=k2, color= 'peru', edgecolor='saddlebrown')
        plt.title(title)
        plt.xlabel(names[i])
        plt.ylabel('Frequency')
        plt.show()
        fig_hist.savefig(title)
        i += 1
    return

def viewhistogram(data, kde=True, name = ""):
    # takes a panda series and creates a histogram of it

    if name != "":
        name = name.rstrip()
        name = name
    else:
        name = data.name

    n = len(data)
    k2 = int(np.ceil(2 * ( n ** (2/5))))   # variable bin width method

    title = 'Histogram of ' + name
    # fig_hist = plt.figure()
    fig_hist = plt.figure(figsize=(8, 4))

    sns.histplot(data, kde=kde, bins=k2, color= 'peru', edgecolor='saddlebrown')
    plt.title(title)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.show()
    fig_hist.savefig(title)
    return


def create_formula(ydat, Xdat):

    # creates a patsyphrase formula for use in Variance Inflation Factors
    # and in the OLS version presented in smf

    if isinstance(ydat, str) : response = ydat
    if isinstance(ydat, pd.DataFrame):
        response = ydat.columns.values[0]
    if isinstance(ydat, pd.Series):
        response = ydat.name

    if isinstance(Xdat, pd.DataFrame):
        selected = Xdat.columns.values.tolist()
    else:
        selected = Xdat

    if response in selected: selected.remove(response)

    formula = "{} ~ {}".format(response,' + '.join(selected))

    return formula


#--------------------------------------------------------------------
#
#   Feature Selection functions
#
#--------------------------------------------------------------------


def powerset(iterable):
    # print('....In powerset')
    from itertools import chain, combinations
    s=list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(1, len(s)+1))

def allcombos(y_data, x_data, sort="AIC"):

    # allow the formulas to print out properly if there are a lot of options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if sort not in ["AIC","BIC","AdjR2",'All'] : sort = 'AIC'
    # print('...In all combos')

    mylist=list(powerset(list(x_data.columns)))
    mylist=[list(row) for row in mylist]

    print('There are {:,} combinations.'.format(len(mylist)))

    ##Target is AIC
    scores=pd.DataFrame(columns=["AIC","BIC","AdjR2","Formula"])

    print('Models complete: ', end=" ")
    for i in range(len(mylist)):
        model = sm.OLS(y_data, x_data[mylist[i]]).fit()
        scores.loc[i,'AIC'] = model.aic
        scores.loc[i,'BIC'] = model.bic
        scores.loc[i,'AdjR2'] = model.rsquared_adj
        scores.loc[i,'Formula'] = create_formula(y_data, x_data[mylist[i]])
        print("{} ".format(' '+str(i+1)), end=" ")
    else:
        print('\n')
    
    top = 5
    if sort == 'All':
        sortgroup = ["AIC","BIC","AdjR2"]
    else:
        sortgroup = [sort]

    for sort_type in sortgroup:
        print('The top {} models, sorted by {}, are:'.format(top, sort_type))
        print(scores.sort_values(by=sort_type, ascending=False).head(top),"\n")

    return

def choose_feature_forward_selected(x_data,y_data):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """


    frames =[y_data, x_data]
    data = pd.concat(frames, axis=1).reindex(y_data.index)

    response = y_data.name

    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {}".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    print('\n{:-^50}'.format('Forward Selected Formula'))
    print(formula)
    return model


def choose_feature_rfe(X, y, num:int=7, passes:int=10):
    
    # Since there can be some differences in the RFE feature selection between
    # two runs of the same data, this function will run the feature selection
    # n times (default: 10) to smooth out the variation between runs.
    # The top k (default: 7) features are selected as a numpy array.
    
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeRegressor
    
    total_ranks = []
    for i in range(X.shape[1]) : total_ranks.append(0)
    for i in range(passes):
        rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=num)
        rfe = rfe.fit(X, y.values.ravel())
        total_ranks = np.add(total_ranks,rfe.ranking_)
        
    adj_ranks = np.divide(total_ranks, passes)
    col_name = X.columns.values

    df_summary = pd.DataFrame({"Feature":col_name, 'Total Ranks':total_ranks, 'Adj Rank': adj_ranks, 'Selected': False})
    df_summary['Rel %'] = round(1 - (df_summary['Total Ranks']-passes)/df_summary['Total Ranks'].max(),3)
    
    df_summary = df_summary.sort_values(by='Adj Rank').reset_index(drop=True)
    
    for i in range(num): df_summary.at[i,'Selected'] = True
    print('\n{:-^50}'.format('Recursive Feature Elimination'))
    print('{:-^50}'.format(' after '+str(passes)+' iterations '))
    print(df_summary.sort_values(by='Adj Rank').to_string(index=False, formatters={'Rel %': '{:,.1%}'.format}))
    
    df_chosen = df_summary.query('Selected == True')
    chosen_features = df_chosen['Feature'].to_numpy()
    
    return chosen_features

def choose_feature_kbest(X, y, func, num:int):
	#### Select K Best

	# K-Best doesn't seem to like the const column, so removing it if it exists
	if 'const' in X.columns : X = X.drop('const', axis=1)

	# feature extraction
	test = SelectKBest(score_func=func, k=num)
	fit = test.fit(X, y)

	#summarize scores
	col_name = X.columns.values

	df_summary = pd.DataFrame({'Feature':col_name,'Selected':fit.get_support(),'Score':fit.scores_})
	df_summary['Rel %'] = round((df_summary['Score'])/df_summary['Score'].max(),3)

	print('\n{:-^50}'.format('K-Best Rankings using ' + func.__name__))    
	print(df_summary.sort_values(by='Score',ascending=False).to_string(index=False, formatters={'Rel %': '{:,.1%}'.format}))

	df_chosen = df_summary.query('Selected == True')
	chosen_features = df_chosen['Feature'].to_numpy()

	return chosen_features

def choose_feature_etc(X, y, cutoff=0.5, passes:int=10):
	### Extra Trees Classifier
	
	total_score = []
	for i in range(X.shape[1]) : total_score.append(0)
	for i in range(passes):
		model = ExtraTreesClassifier(n_estimators=X.shape[1])
		model.fit(X, y)
		total_score = np.add(total_score,model.feature_importances_)
		
	col_name = X.columns.values
	
	df_summary = pd.DataFrame({'Feature':col_name,'Score':total_score, 'Norm Score':np.divide(total_score,passes)})

	df_summary['Rel %'] = round((df_summary['Score'])/df_summary['Score'].max(),3)
	df_summary['Selected'] = df_summary['Rel %'] >= cutoff

	print('\n{:-^50}'.format('Extra Trees Classifier Scoring'))
	print('{:-^50}'.format(' after '+str(passes)+' iterations '))
	print(df_summary.sort_values(by='Score',ascending=False).to_string(index=False, formatters={'Rel %': '{:,.1%}'.format}))
	
	df_chosen = df_summary.query('Selected == True')
	chosen_features = df_chosen['Feature'].to_numpy()
	
	return chosen_features

def choose_feature_lasso(X, y, alpha=1):

    col_name = X.columns.values
    
    #X[col_name]	= X[col_name]/X[col_name].max()
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    
    X_std = ss.fit_transform(X)
    y_std = ss.fit_transform(y[:, np.newaxis]).flatten()

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_std, y_std)
    

    df_summary = pd.DataFrame({'Feature':col_name, 'Coef':lasso.coef_, 'Abs Coef':abs(lasso.coef_)})
    df_summary['Rel %'] = round((df_summary['Abs Coef'])/df_summary['Abs Coef'].max(),3)
    df_summary['Selected'] = df_summary['Coef'] != 0
    
    header = 'Lasso Regression Score: {:.6f}'.format(lasso.score(X_std, y_std))
    print('\n{:-^50}'.format(header))  
    print(df_summary.sort_values(['Selected','Abs Coef'],ascending=False).to_string(index=False, formatters={'Rel %': '{:,.1%}'.format}))
    
    df_chosen = df_summary.query('Selected == True')
    chosen_features = df_chosen['Feature'].to_numpy()

    return chosen_features

def choose_feature_ridge(X, y, alpha=1, cutoff=0):

    col_name = X.columns.values
    
    #X[col_name]	= X[col_name]/X[col_name].max()

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    
    X_std = ss.fit_transform(X)
    y_std = ss.fit_transform(y[:, np.newaxis]).flatten()

    rr = Ridge(alpha=alpha)
    rr.fit(X_std, y_std)
	
    df_summary = pd.DataFrame({'Feature':col_name,'Coef':rr.coef_, 'Abs Coef':abs(rr.coef_)})
    df_summary['Rel %'] = round((df_summary['Abs Coef'])/df_summary['Abs Coef'].max(),3)
    df_summary['Selected'] = df_summary['Abs Coef'] >= cutoff
    
    header = 'Ridge Regression Score: {:.6f}'.format(rr.score(X_std, y_std))
    print('\n{:-^50}'.format(header))    
    print(df_summary.sort_values(by='Abs Coef',ascending=False).to_string(index=False, formatters={'Rel %': '{:,.1%}'.format}))

    df_chosen = df_summary.query('Selected == True')
    chosen_features = df_chosen['Feature'].to_numpy()

    return chosen_features

#--------------------------------------------------------------------
#
#                 Model Evaluation Functions
#      
#--------------------------------------------------------------------
# functions here either print out analytic results of tests on 
# data and the residuals, or display and save off graphics

#--------------------------------------------------------------------
def VIFcheck(patsyphrase, data):
    #VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from patsy import dmatrices

    # Break into left and right hand side; y and X
    _, X = dmatrices(patsyphrase, data=data, return_type="dataframe")
    # For each Xi, calculate VIF
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    print('\n{:=^70}'.format(""))
    print('{:^70}'.format("Variance Inflation Factors"))

    print(vif)
    border('-')

    return

#--------------------------------------------------------------------
def crossvalidation(y_train, y_test, X_train, X_test, my_model):

    from sklearn.metrics import mean_squared_error

    y_pred_train = my_model.predict(X_train)
    y_pred_test = my_model.predict(X_test)

    keys = ['RMSE training:','RMSE test:', 'Model Cross Validation Score:']

    RMSE_train = mean_squared_error(y_train, y_pred_train)

    RMSE_test = mean_squared_error(y_test, y_pred_test)

    RMSE_mcv = RMSE_test - RMSE_train

    print('\n{:=^70}'.format(""))
    print('{:^70}\n'.format("Model Cross Validation"))

    dictionary = dict(zip(keys,[RMSE_train, RMSE_test, RMSE_mcv]))
    print("\n".join("{:>30} {:<15}".format(k, v) for k, v in dictionary.items()))
    print('')
    border('=')

    return

#--------------------------------------------------------------------
def analyzeresiduals(model):

    from statsmodels.stats.stattools import durbin_watson
    from scipy.stats import shapiro
    import statsmodels.stats.api as sms

    # displays results for Assumptions 2, 4, 5
    # all of which deal with the residuals of the model

    #Pull Residuals
    Residuals = model.resid

    # ASSUMPTION 2
    # HO = Gaussian / normal distribution (good)
    # HA = Not a normal distribution (bad)
    print('{:=^70}'.format(""))
    print('{:^70}'.format("Assumption - Normality of Residuals"))
    print('{:^70}\n'.format("Shapiro-Wilk Test on Residuals"))
    # Shapiro-Wilk Test
    keys = ['Shapiro-Wilk statistic:',
            'SW test\'s p-value:']

    results = shapiro(Residuals)
    
    dictionary = dict(zip(keys, results))
    print("\n".join("{:>35} {:<15}".format(k, v) for k, v in dictionary.items()))

    # interpret
    alpha = 0.05
    if results[1] > alpha:
        print('\n + Residuals looks Gaussian (fail to reject H0)')
    else:
        print('\n ! Residuals does not look Gaussian (reject H0)')

    # Check centered at zero by looking at the mean
    #Number returned should be near zero
    print('{:-^70}'.format(""))
    print('{:^70}\n'.format("Residuals mean check"))
    print("{:>35} {:<15}".format('Residuals centered at zero:',np.mean(Residuals)))
    border('-')

    # ASSUMPTION 4
    print('{:=^70}'.format(""))
    print('{:^68}\n'.format('Assumption - Variance of the Errors is Constant'))
    # HO = homoscedasticity (good)
    # HA = heteroscedasticity (bad)
    keys = ['Lagrange Multiplier statistic:',
            'LM test\'s p-value:',
            'F-statistic:', 
            'F-test\'s p-value:',
            ]
    results = sms.het_breuschpagan(Residuals, model.model.exog)
    
    dictionary = dict(zip(keys, results))
    print("\n".join("{:>35} {:<15}".format(k, v) for k, v in dictionary.items()))

    # interpret
    alpha = 0.05
    if results[3] > alpha:
        print('\n + Residuals show homoscedasticity (fail to reject H0)')
    else:
        print('\n ! Residuals show heteroscedasticity ! (reject H0)')
    border('-')

    # ASSUMPTION 5
    print('{:=^70}'.format(""))
    print('{:^70}\n'.format('Assumption - Errors are Independent'))
    # HO = no serial correlation.  No pvalue returned however.
    # DW stat ranges 0 to 4, centered on 2, with 2 showing independence
    # 0-2: positive serial correlation
    # 2-4: negative serial correlation
    # A number close to "2" shows independence.  
    # Values far from 2 indicate autocorrelation of residuals (bad).
    dw = durbin_watson(Residuals)
    autocorr = (2-dw)/2
    print('{:>35} {:<15}\n'.format('Durbin Watson test statistic:',dw))
    if autocorr > 0:
        print('> It shows a {:.1%} positive autocorrelation'.format(autocorr))
    elif autocorr < 0:
        print('> It shows a {:.1%} negative autocorrelation'.format(abs(autocorr)))
    else:
        print('+ It shows complete independence.')
    
    border('=')

    return

#--------------------------------------------------------------------
def examineresiduals(model, name:str="", **kwargs):
    '''
    Plots out mutiple graphs and plots to examine the residuals of a model.

    Parameters:
    
    model : an OLS model
    
    name (str) : (optional) Will be added to plot titles and filenames
    
    **kwargs : allow for customization of the various plots. Recongized variables are:
         bars (bool) : if bars is true, vertical grid lines added to Residuals vs Fitted Values
         color : Mathplotlib color value. Plot default is orchid
     edgecolor : Mathplotlib color value. Plot default is darkorchid
        marker : Mathplotlib marker style. Plot default is dot '.'

    Returns:
    Will display and save off five plots for graphic analysis of residuals. Doesn't return data to the caller.
    1. Residuals vs Fitted Values:
        The model's residuals against the model's fitted values
        Options: 'color', 'edgecolor', 'marker', 'bar' **kwargs will affect this plot
        If the bar kwarg is passed, will add two vertical lines to plot, breaking it into equal vertical thirds.
    2. Histogram and Box plot:
        Displays the two plots as a single figure for normallity check on residuals
        Uses sns.histplot, with kde=True, so returns a fitted curve over the histogram
        Options: 'color' and 'edgecolor' **kwargs will affect this plot
    3. QQ-plot of residuals
        Options: 'marker' and 'edgecolor' **kwargs will affect this plot
    4. Standardized Residuals vs Fitted Line to check for outliers
        Options: 'color' and 'marker' **kwargs will affect this plot
    5. Influence plot showing H Leverage against the studentized residuals
        No options will change color scheme layout
    '''


    name = name.rstrip()
    name = name + "_"

    # set up some defaults and check kwargs for customization
    dft_color = 'orchid'
    dft_color2 = 'darkorchid'
    dft_marker = '.'
    gridit = False

    if 'color' in kwargs:
        color1 = kwargs.get("color")
    else:
        color1 = dft_color
    if 'edgecolor' in kwargs:
        color2 = kwargs.get('edgecolor')
    else:
        color2 = dft_color2
    if 'marker' in kwargs:
        marker = kwargs.get('marker')
    else:
        marker = dft_marker
    if 'bars' in kwargs:
        gridit = True

    # get x axis - the Fitted values
    Fitted_Values = model.fittedvalues
    
    # now Y axis - of different flavors
    #Pull Residuals
    Residuals = model.resid

    # standardized residuals
    results = model.model.fit()
    influence = results.get_influence()
    Std_Resid = influence.resid_studentized_internal

    # Studentized redisidual. Just in case.
    # Stud_Resid = influence.resid_studentized_external

    # Now the graphs

    #Plot Residuals vs Fitted Values
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(Fitted_Values, Residuals, alpha=1.0, color=color1, marker=marker)
    fig.suptitle('Residuals versus Fitted Values '+name[:-1]+' Model')
    ax.set_title(" ")
    plt.ylabel("Residual")
    plt.xlabel("Fitted Values")
    fig.tight_layout(pad=2)
    ax.grid(True)
    plt.axhline(y=0,color='black')
    if gridit:
        plt.axvline(x=min(Fitted_Values)+(max(Fitted_Values)-min(Fitted_Values))/3, color=color2)
        plt.axvline(x=min(Fitted_Values)+2*(max(Fitted_Values)-min(Fitted_Values))/3, color=color2)
    fig.savefig(name+'Residuals_fig1.png', dpi=300)

    #Check for normality distribution of residuals
    fig2, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(12,8))
    sns.boxplot(data=Residuals, color=color2, ax=ax_box)
    sns.histplot(Residuals, kde=True, ax=ax_hist, color=color1, edgecolor=color2)
    #fig2.suptitle(name[:-1]+" Model Residuals Normality Ceck")
    ax_box.set_title(name[:-1]+" Model Residuals Normality Check")
    ax_box.set(xlabel='Residuals')
    fig2.savefig(name+'Residuals_fig2_Histogram.png', dpi=300)

    #Check for normality distribution of residuals - QQ plot
    pp = sm.ProbPlot(Residuals, stats.norm, fit=True)
    fig3 = pp.qqplot(marker='.', markerfacecolor=color2, markeredgecolor=color2, alpha=0.8)
    fig3.suptitle(name[:-1]+" Model Residuals as QQ-Plot Normality Check")
    sm.qqline(fig3.axes[0], line='45', fmt='k--')
    fig3.savefig(name+'Residuals_fig3_QQplot.png', dpi=300)
    
    #Plot Standard Residuals vs Fitted Values
    fig4, ax = plt.subplots(figsize=(12,8))
    ax.scatter(Fitted_Values, Std_Resid, alpha=1.0, color=color1, marker=marker)
    fig4.suptitle('Standard Residuals versus Fitted Values for '+name[:-1]+' Model')
    ax.set_title(" ")
    plt.ylabel("Standard Residual")
    plt.xlabel("Fitted Values")
    fig4.tight_layout(pad=2)
    ax.grid(True)
    plt.axhline(y=0, color='black')
    plt.axhline(y= +3, color='xkcd:cool grey', linestyle = '--')
    plt.axhline(y= -3, color='xkcd:cool grey', linestyle = '--')
    fig4.savefig(name+'Residuals_fig4_Standard.png', dpi=300)

    #Influence Plot
    fig5, ax = plt.subplots(figsize=(12,8))
    fig5 = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    ax.set_title(name[:-1]+" Model Influence Plot")
 
    fig5.savefig(name+'Residuals_fig5_Influence_Plot.png', dpi=300)

    plt.show()

    return

#--------------------------------------------------------------------
def fittedline(y_train, my_model, name:str = ""):
    '''
    Returns a graphic with the fitted line plotted against a scatter plot
    of the model's fitted values vs the observed values in y_train. Additionally,
    confidence bands for predicted and observed are plotted along the fitted line.

    Parameters:
    y_train (nparray, pd.Series): The observerd / response / dependant variable
    
    my_model : an OLS model
    
    name (str): Optional. Added to title of graphic and filename.  Graphic's default
    name is 'Model Fit Plot'.

    Returns:
    Saves off graphic titled either Fitted Values.png or name + Fitted Values.png, if
    name provided.
    Also diplays the graphic.
    Doesn't return a variable otherwise.
    '''

    if name != "" : name = name + " "
     
    Fitted_Values = my_model.fittedvalues

    df_Fitted_Values=pd.DataFrame(Fitted_Values, columns=['Fitted_Values']) 

    frames =[y_train, df_Fitted_Values]
    newdf = pd.concat(frames, axis=1).reindex(y_train.index)

    if isinstance(y_train, pd.DataFrame):
        y_name = y_train.columns.values[0]
    if isinstance(y_train, pd.Series):
        y_name = y_train.name

    patsyphrase = y_name + '~Fitted_Values'
    model=smf.ols(patsyphrase, newdf).fit()

    Beta0=model.params[0]
    Beta1=model.params[1]

    fitted = lambda xx: Beta0+Beta1*xx

    x_pred = np.linspace(newdf['Fitted_Values'].min(),newdf['Fitted_Values'].max(),200)
    y_pred = fitted(x_pred)

    dfpred = pd.DataFrame({'Fitted_Values':x_pred, y_name:y_pred})
    prediction = model.get_prediction(dfpred)
    predints = prediction.summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(newdf['Fitted_Values'], newdf[y_name], alpha=1.0, color='xkcd:emerald green', marker='d')
    fig.suptitle(name +'Model Fit Plot')
    plt.xlabel("Fitted Values")
    plt.ylabel('Observed Values')
    fig.tight_layout(pad=2)
    ax.grid(True)
    ax.plot(x_pred, y_pred, '-', color='xkcd:cranberry', linewidth=2)

    ax.fill_between(x_pred, predints['mean_ci_lower'], predints['mean_ci_upper'], color='#888888', alpha=0.6)
    ax.fill_between(x_pred, predints['obs_ci_lower'], predints['obs_ci_upper'], color='#888888', alpha=0.2)
    fig.savefig(name + 'Fitted Values.png', dpi=300)
    plt.show()

    return

#--------------------------------------------------------------------#
#                                                                    #
#                         End Functions                              #
#                                                                    #
#--------------------------------------------------------------------#