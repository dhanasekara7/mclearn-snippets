```python
# 1)What is the average age (age feature) of women?
data.loc[data['sex'] == 'Female', 'age'].mean()
# or
data[data["sex"]=="Female"]["age"].mean()

df['Churn'] = df['Churn'].astype('int64')

#　include the datatype in describe
df.describe(include=['object', 'bool'])

df['Churn'].value_counts()

# to show in fractions
df['Churn'].value_counts(normalize=True)

# sorting
df.sort_values(by=['Churn', 'Total day charge'],
        ascending=[True, False]).head()

# mean
df['Churn'].mean()

df[df['Churn'] == 1].mean()

#What is the maximum length of international calls among loyal users (Churn == 0) who do not have an international plan?
df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max()

#f we need the first or the last line of the data frame, we can use the df[:1] or df[-1:]


#Applying Functions to Cells, Columns and Rows¶
df.apply(np.max)

# something starts with
df[df['State'].apply(lambda state: state[0] == 'W')].head()

# map to other values
d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
# or
df.replace({'International plan':d})


# group by
columns_to_show = ['Total day minutes', 'Total eve minutes',
                   'Total night minutes']

df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])

# summary table

#cross tab
pd.crosstab(df['Churn'], df['International plan'])

pd.crosstab(df['Churn'], df['International plan'], normalize=True)

# summary also
pd.crosstab(df['Churn'], df['International plan'], margins=True)


# pivot table
df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
               ['Area code'], aggfunc='mean')

# get rid of columns
df.drop(['Total charge', 'Total calls'], axis=1, inplace=True)

# and here’s how you can delete rows
df.drop([1, 2]).head()

# sns countplot
sns.countplot(x='International plan', hue='Churn', data=df);


# some x value gt some value
df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')

pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True)

sns.countplot(x=Many_service_calls', hue='Churn', data=df)

# & crosstab
pd.crosstab(df['Many_service_calls'] & df['International plan'] , df['Churn'])

# size of dataset
len(data)
data.shape(0)

# find education whose salary above 50K
data.loc[data['salary'] == '>50K', 'education'].unique()

#Among whom the proportion of those who earn a lot(>50K) is more:
# among married or single men (marital-status feature)?
# Consider married those who have a marital-status starting with Married
# (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse),
# the rest are considered bachelors.
data[ (data["sex"] == "Male") &
      (data["marital-status"].isin(['Never-married',
                                      'Separated',
                                      'Divorced',
                                      'Widowed'])), "salary"].value_counts()

data[ (data["sex"] == "Male") &
      (data["marital-status"].str.startswith('Married')), "salary"].value_counts()

# for and group by
for (country, salary), sub_df in data.groupby(['native-country', 'salary']):
    print(country, salary, round(sub_df['hours-per-week'].mean(), 2))

pd.crosstab(data['native-country'], data['salary'],
           values=data['hours-per-week'], aggfunc=np.mean)

# 9. What is the maximum number of hours a person works per week (hours-per-week feature)?
max_load = data['hours-per-week'].max()

# How many people work such a number of hours and
num_workaholics = data[data['hours-per-week'] == max_load].shape[0]

# what is the percentage of those who earn a lot among them?
rich_share = float(data[(data['hours-per-week'] == max_load)
                 & (data['salary'] == '>50K')].shape[0]) / num_workaholics
```