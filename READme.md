Regression Project - Estimating Zillow Property Prices 

Jerry Nolf - Codeup - Innis Cohort - March 2022  

----  
## PROJECT OVERVIEW  

 
### 1.  Goals:
The goal of this project is to identify key drivers that will help to estimate property prices based on available Zillow data. With drivers identified, a model will be built in order to predict the tax value of single family properties purchased in 2017. In the end, the following deliverables will be available:

- This README file - provides an overview of the project and steps for project reproduction.  
- Draft Jupyter Notebook - provides all steps taken to produce the project.
- acquire.py - used to acquire data
- prepare.py - used to prepare data
- Report Jupyter Notebook - provides final presentation-ready assessment and recommendations.   

---- 
### 2. DESCRIPTION:

This project will begin by acquiring the appropriate data from databases on the mySQL server. The data will then be prepared for exploration by dropping null entries and removing outliers so that the data is usable for a machine learning regression model. After the data is prepared that exploration phase will allow us to dive deeper and identify key drivers of tax value. This exploration will provide more knowledge of the data and initial question.

---- 
#### INITIAL QUESTIONS: 

- What county has the highest tax values?
- Does house square footage affect tax value?
- Does number of bedrooms affect tax value?
- Does number of bathrooms affect tax value? 

----  
## DATA DICTIONARY:

The final DataFrame used to explore the data for this project contains the following variables (columns).  The variables, along with their data types, are defined below:  


|   Column_Name   | Description | Type      |
|   -----------   | ----------- | ---------- |
| bathrooms | Number of bathrooms | float |
| bedrooms   | Number of bedrooms | int64  |
| sqft      |  Calculated total finished living area in square feet   | int64 |
| lot_sqft      |  Area of the lot in square feet | int64 |
| tax_value   | The total tax assessed value of the parcel       | int64    | 
| county   | The county in which the property is located  |  object |


---- 
## PROCESS:
The following outlines the process taken through the Data Science Pipeline to complete this project.  

Plan ➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver

### 1. PLAN
- Define the project goal
- Determine proper format for the audience
- Asked questions that would lead to final goal
- Define an MVP


### 2. ACQUIRE
- Create a function to pull appropriate information from the zillow database
- Create and save an acquire.py file and made it available to import


### 3. PREPARE
- Ensure all data types are usable
- Rename columns to improve readability
- Rename entries to a more informative names
- Add a function that splits the acquired data into Train, Validate, and Test sets
- 20% is originally pulled out in order to test in the end
- From the remaining 80%, 30% is pullout out to validate training
- The remaining data is used as testing data
- In the end, there should be a 56% Train, 24% Validate, and 20% Test split 
- Create a prepare.py file with functions that will quickly process the above actions


### 4. EXPLORE
- Create an exploratory workbook
- Create initial questions to help explore the data further
- Make visualizations to help identify and understand key drivers
- Use stats testing on established hypotheses


### 5. MODEL & EVALUATE
- Use models to evaluate true drivers of assessed tax value
- Create a baseline
- Make predictions of models and what they say about the data
- Compare all models to evaluate the best for use
- Use the best performing model on the test (unseen data) sample
- Compare the modeled test versus the baseline


### 6. DELIVERY
- Present a final Jupyter Notebook
- Make modules used and project files available on Github

 ---- 
## REPRODUCIBILITY: 
	
### Steps to Reproduce
1. Have your env file with proper credentials saved to the working directory

2. Ensure that a .gitignore is properly made in order to keep privileged information private

3. Clone repo from github to ensure availability of the acquire and prepare imports

4. Ensure pandas, numpy, matplotlib, scipy, sklearn, and seaborn are available

5. Follow steps outline in this README.md to run Final_Zillow_Report.ipynb


---- 
## KEY TAKEAWAYS:

### Conclusions
The goals of this project were to identify key drivers of tax value for single residential purchased during 2017. These key drivers were found to be the following:
        
    - House Square Footage
    - Number of Bedrooms
    - Number of Bathrooms

Using these drivers, along with lot square footage, we're able to build a model that is expected to perform with an RMSE of $208,423 on unseen data.
### Recommendations
- Number of bedrooms and bathrooms along with house and lot square footage can be used to help improve the performance of a machine learning regression model.

- While our model does work to improve on our baseline, higher quality data is needed in order to fine-tune our model. Our new model used data that had to drop 19,800 rows in order to had true usable data.
### Next Steps
With more time, I would like to:
- Work on more feature engineering and explore relationship of categories to tax value further.
- Gather more adequate and complete data that will allow for a clearer picture and the possibility for a more refined and detailed model.  

---- 
