# Machine-Learning-fundations
This repositiory is for the Machine Learning fundations at my university. Here I will develop my first project for this course.

Name of student: Laura Somogyi

Course: Machine Learning foundation

Semester: 2025-2026 2nd semester

# Task 1 -> Identifying the Prediction Target

*   Inspect the dataset and identify which column should be treated as the target variable for this assignment.
*   Justify why this column represents the appropriate prediction objective in the context of the marketing campaign


*   Identify at least two other variables that could superficially appear to be valid targets and explain why they should not be treated as the prediction objective

My answers

* The column that should be treated as a target variable is the "y" column since it adresses the question whether the client will subscribe to the term deposit or not.

*  It is appropriate prediction objective in the context of the marketing campaign because as I mentioned above the subscribed or not will be marked here and this decides whether the bank gets a share from penalties or not or they manage to get a client or not.

* potential other target variables
    * "poutcome" column
        * This feature represents whether the previous campain was successful, failed or none-existent.
        * This could be a reasonable if we had information about the previous interviews like duration and when it took place. However the reason I did not pick this is it may be correlated with the one included in this dataset. So the outcome may include the effect of both. 
    * "balance" column
        * It could be a possible target because it keeps track of the average yearly balance on the term deposit account and this can be predicted from many variables.
        * However I will not treat this as a prediction objective because it is not available at the time when we want to predict whether the client will commit to the creation of the account or not. Until then the balance is either 0 or missing or simply just unknown
     

# Task 2 Task Ordering

*   Lecture material:
    *   Lecture 2 (Data Splitting and Leakage), Lecture 5 (Preprocessing), Lecture 9 (ML Pipeline)
*   Determine the correct order in which the data preparation tasks in this assignment should be performed
*   Provide a structured justification for your chosen order
*   For each step in your proposed sequence, explain:
    *   what information is allowed to be used at that stage;
    *   what information must not be used;
    *   what type of data leakage could occur if the order were changed
*   discuss at least one example of an incorrect ordering and explain the consequences it would have on model evaluation

My answer

For the order I have decided on the one which can be seen in thes jupyter notebook for the following reasons:
    * First I chose to identify the prediction target, which gave me an idea which way should I start my assignment. Then I chose task ordering, to structure my tasks according to the target variable and what seemed reasonable to work with the data.
    * Then the reason I chose data exploration as 3rd is to verify whether I can go on with my 

I will provide the rest of explanation at each task part.


# Task 3: Data Loading and Exploration
Lecture material: Lecture 1 (Problem Formulation), Lecture 2 (Data Inspection and EDA).

• Load the dataset into a Pandas DataFrame.

• Inspect the structure of the dataset: number of observations, number of features, data types, and basic
summary statistics.

• Identify which variables are numerical and which are categorical

 Analyze the distribution of the target variable and comment on potential class imbalance.

• Detect explicit and implicit missing values (e.g., special categories such as unknown).

• Visualize the distribution of at least:

    – two numerical variables; and
    – two categorical variables.


• Identify at least one variable that may require special consideration before modeling (e.g., due to distributional
properties, extreme skewness, or availability at prediction time), and briefly justify your reasoning.

Note: Exploratory analysis is not a checklist of plots. Each visualization or statistic should support a specific
observation or hypothesis about the data

My explanation:
* numerical variables
    * age
    * deafult
    * balance
    * housing
    * loan
    * duration
    ÷ campaign
    * pdays
    * previous    
* categorycal variables
    * poutcome
    * contact
    * education
    * marital
    * job
* date type variables
    * month
    * day_of_week

* distribution of the target variable and comment on pototential class imbalance
    * By taking a look at the barblots in my code it is visible for me that 36,548 (88.7%) from the contacted people did not sign the contract with the bank.
    * On the other side 4,640 (11,3%) of contacted people decided on signing up for the subscription offered by the bank.
    * This implies a quite large class imbalance in the subcription acception rate. Seemingly the majority did not accept the contract and the rest did.
    * It is very much a skewed distribution, more in detail right skewed.

* missing values
    * explicit
        * poutcome
        * default
    * implicit
        * previous
        * marital status
        * education

* one variable that may require special consideration before modeling
    * Maybe it is the job variable, because it very much effects whether a person is eligible for a contract or not. This is also corrolated with income since I assume the person has to pay for the contract and wage depends on the job type. Therefore there are income differences and other variables can be assumed from this. For example the person had a loan or whether the person defaulted or not.
 
* plot usage explanation
   * For subscription distribution I used the barplot because it represents 2 values compared to each other very nicely. Also, I have included some numerical values and percentages so it provides those informations visually too.
   * For age distribution I have used both gaussian kde plot which is a non-parametric, smooth, continuous curve used to visualize the probability density function (PDF) of a dataset and barplot. The barplot is because it represents the distribution including every age as a bar which holds a lot more information. On the other side the gaussian kde function is a little more simplified and shows us well the distribution of ages which is nearly normal but a bit more right skewed.
   * For representing the job I used a barplot because is separates the distinct jobs and show the number of people working in that job and also showing a distribution if we look at the whole diagram. Also for the relationship status is for the same reason that it is much more informative as a barplot than any other.    

AI prompt disclaimer:

- prompt -> "How do I plot a binary target variable in python from a CSV file?"
* reply:
    * Here's a complete guide to plotting a binary target variable:
    * Count Plot (Most Common for Binary Targets)
    * sns.countplot(x='target', data=df)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    * Bar Chart with Percentages
    * counts = df['target'].value_counts()
    percentages = df['target'].value_counts(normalize=True) * 100

    plt.bar(counts.index, counts.values, color=['steelblue', 'salmon'])
    plt.title('Target Variable Distribution')
    plt.xlabel('Class (0 = Negative, 1 = Positive)')
    plt.ylabel('Count')

    Add percentage labels on bars
    for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
        plt.text(i, count + 5, f'{pct:.1f}%', ha='center', fontweight='bold')

    plt.show()
    * Pie Chart
    df['target'].value_counts().plot(
    kind='pie',
    labels=['Negative (0)', 'Positive (1)'],
    autopct='%1.1f%%',
    colors=['steelblue', 'salmon']
    )
    plt.title('Target Variable Distribution')
    plt.ylabel('')
    plt.show()

    * All plots together
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    Count plot
    sns.countplot(x='target', data=df, ax=axes[0])
    axes[0].set_title('Count Plot')

    Bar chart
    counts = df['target'].value_counts()
    axes[1].bar(counts.index, counts.values, color=['steelblue', 'salmon'])
    axes[1].set_title('Bar Chart')

    Pie chart
    df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[2])
    axes[2].set_title('Pie Chart')

    plt.tight_layout()
    plt.show()

    * Install dependencies if needed: pip install pandas matplotlib seaborn
    * Replace 'target' with your actual column name and 'your_file.csv' with your file path. The count plot or bar chart is generally recommended for binary targets as they make class imbalance immediately obvious.


- 2nd prompt -> "How do I print out more data from a CSV file in python?"

ANSWER I GOT

Here are the different ways to print more data from a CSV file:

1. Load the Data
pythonimport pandas as pd

df = pd.read_csv('your_file.csv')

2. Print More Rows
python# Default only shows 5 rows
print(df.head())

how more rows
print(df.head(20))        # first 20 rows
print(df.tail(20))        # last 20 rows
print(df.head(100))       # first 100 rows

3. Print All Rows & Columns (No Truncation)
pythonpd.set_option('display.max_rows', None)       # show all rows
pd.set_option('display.max_columns', None)    # show all columns
pd.set_option('display.width', None)          # no line wrapping
pd.set_option('display.max_colwidth', None)   # full column content

print(df)

4. Print Specific Columns
python# Single column
print(df['column_name'])

Multiple columns
print(df[['col1', 'col2', 'col3']])

5. Print Specific Rows
python# By index range
print(df[0:50])           # rows 0 to 49

By condition
print(df[df['age'] > 30])

By specific index
print(df.iloc[10])        # single row
print(df.iloc[10:20])     # row range

6. Useful Summary Prints
pythonprint(df.shape)           # (rows, columns)
print(df.columns)         # all column names
print(df.dtypes)          # data types of each column
print(df.describe())      # stats summary (count, mean, std, etc.)
print(df.info())          # overview of the dataframe
print(df.nunique())       # unique values per column
print(df.isnull().sum())  # missing values per column

7. Reset Options Back to Default
pythonpd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

Quick Reference:
GoalCodeFirst N rowsdf.head(N)Last N rowsdf.tail(N)All rowspd.set_option('display.max_rows', None)Specific columnsdf[['col1', 'col2']]Filter rowsdf[df['col'] > value]Summary statsdf.describe()
For your bank dataset specifically, since it uses semicolons as separators, remember to load it with:
pythondf = pd.read_csv('bank-additional-full.csv', sep=';')

# Task 4 Managing Missing Values

Lecture material: Lecture 2 (Data Inspection), Lecture 5 (Preprocessing and Pipeline Discipline).

• Identify both explicit missing values (e.g., NaN) and implicit missing values (e.g., categories such as unknown or sentinel numerical values, i.e., values that may represent special codes rather than genuine measurements).

• Quantify the extent of missingness for each affected variable.

• Propose and justify a strategy for handling missing values in each case (e.g., removal, imputation, separate
category, indicator variable).

• Clearly state which operations must be fitted using the training set only, and explain why.
Note: Your strategy should distinguish between “data cleaning” decisions (e.g., correcting inconsistent entries)
and “modeling” decisions (e.g., whether missingness itself may carry predictive information).

My answer
* explicit missing value
   * poutcome 
* implicit missing value
   * default
   * housing
   * loan
* quantifying value missingness is in my code document
* My strategy for handling missing values is keeping them in the data set, but marking them with a different value. As it is visible in my code, I marked them with NA since it is a unique value for the missing values. The reason I would not remove them is because the dataset is not large enough in my judgement so I would rather not remove values which would desort the outcome.
* 

# Task 5: Encoding Categorical Variables

Lecture material: Lecture 4 (Categorical Encoding), Lecture 6 (Linear Models), Lecture 9 (Feature Engineering
and Expressiveness).

• Identify all categorical variables in the dataset.

• Distinguish between nominal variables (categories without intrinsic order, e.g., job type) and ordinal variables
(categories with a meaningful order, e.g., education level), and justify your classification.

• Select and apply an appropriate encoding strategy for each categorical variable.

• Clearly state which encoders must be fitted on the training set only, and explain why.

• Analyze how encoding changes:

    – the dimensionality of the dataset;
    – the interpretability of model coefficients;
    – the types of decision boundaries a linear model can represent.

Note: Encoding is not a purely mechanical transformation. Your justification should explicitly connect your encoding
decisions to the assumptions and behavior of Logistic Regression

# Task 6: Feature Selection
Lecture material: Lecture 5 (Feature Selection), Lecture 6 (Linear Models), Lecture 9 (Pipeline Discipline).

• Identify and remove features with very low variance, if any. Justify the criterion used to define “low” variance.

• Identify highly correlated numerical features and decide whether any should be removed. Clearly state the
threshold used and justify your decision.

• Discuss whether any features should be removed based on conceptual considerations (e.g., redundancy,
availability at prediction time, or risk of data leakage).

• Explain why feature selection must be performed using the training set only.

• Discuss the consequences of performing feature selection on the entire dataset before splitting.
Note: Feature selection is not purely statistical. Your reasoning should explicitly connect your decisions to the assumptions and stability of Logistic Regression

# Task 7: Data Splitting
Lecture material: Lecture 2 (Data Splitting and Leakage), Lecture 9 (ML Pipeline).

• Split the dataset into training, validation, and test sets.

• Justify your choice of proportions for each split.

• Perform stratified splitting with respect to the target variable and explain why stratification is necessary for
this dataset.

• Clearly describe at which stage of your pipeline the split must occur, and explain what types of data leakage
would arise if splitting were performed later.

Note: A recommended strategy is to first split the dataset into a training set and a temporary set, and then
split the temporary set into validation and test sets. Use the stratify argument of train test split where
appropriate

# Task 8 Addressing Class Imbalance

Lecture material: Lecture 3 (Class Imbalance), Lecture 4 (Evaluation Metrics), Lecture 9 (Pipeline Discipline).

• Quantify the class distribution in the training set and explain why imbalance is or is not a concern for this
prediction task.

• Propose and apply a resampling strategy (e.g., random oversampling, SMOTE, or ADASYN). Clearly justify
at which stage of the pipeline the resampling step should occur.

• Justify your choice of resampling method in terms of its assumptions and expected effect on the learning
algorithm.

• Explain what would happen if resampling were applied before splitting the dataset into training, validation,
and test sets. Discuss the implications for model evaluation.

• Briefly discuss how class imbalance affects evaluation metrics such as accuracy, precision, and recall.

Note: Resampling is part of the training procedure and must be applied to the training set only. Validation and
test sets must preserve the original class distribution

# Task 9: Feature Scaling

Lecture material: Lecture 5 (Feature Scaling), Lecture 6 (Logistic Regression and Optimization)..

• Identify the numerical variables that require scaling.

• Select and apply an appropriate scaling strategy (e.g., standardization or normalization) to those variables.

• Justify your choice of scaling method in the context of Logistic Regression.

• Clearly state which transformations must be fitted on the training set only, and explain why.

• Discuss how feature scaling affects

    – gradient-based optimization;
    – the magnitude and comparability of model coefficients;
    – the interpretation of regularization penalties.

Note: Feature scaling is not a cosmetic transformation. Your justification should explicitly connect your scaling
decision to the mathematical behavior of linear models

# Task 10 training a Logistic Regression Model

Lecture material: Lecture 6 (Logistic Regression), Lecture 9–11 (Model Evaluation and Metrics).
• Train a Logistic Regression model to predict whether a client subscribes to a term deposit.
• Use the validation set to generate predictions.
• Report at least Accuracy, Precision, and Recall on the validation set.
• Compare the model’s accuracy with the Zero Rule baseline and briefly interpret the result.
Note: The goal here is not to squeeze out the best possible performance. The goal is to verify that your data
preparation pipeline is coherent and correctly implemented. If your preprocessing is principled, the model should
behave sensibly. If it behaves strangely, that is a signal to revisit earlier decisions. Have fun finding a visually
appealing way to display the predictions or the confusion matrix on the validation set. This is your chance to make
the output readable and professional 8-)
