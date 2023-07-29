### KICKOFF - CODING AN APP IN STREAMLIT

### import libraries
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from PIL import Image
#from sklearn.model_selection import train_test_split
from surprise import Dataset
from surprise.reader import Reader
from surprise.prediction_algorithms.matrix_factorization import SVD as FunkSVD


#######################################################################################################################################
### LAUNCHING THE APP ON THE LOCAL MACHINE
### 1. Save your *.py file (the file and the dataset should be in the same folder)
### 2. Open git bash (Windows) or Terminal (MAC) and navigate (cd) to the folder containing the *.py and *.csv files
### 3. Execute... streamlit run <name_of_file.py>
### 4. The app will launch in your browser. A 'Rerun' button will appear every time you SAVE an update in the *.py file

# Press R in the app to refresh after changing the code and saving here

### To position text and color, you can use html syntax
# st.markdown("<h1 style='text-align: center; color: blue;'>Our last morning kick off</h1>", unsafe_allow_html=True)

#######################################################################################################################################
### App title
st.title("Food Recipe Recommender App")
st.write('This application will help you to find the best recipes based on the similarities with your favorite food.')
st.write('NOTE: Due to file size limitations, a very small size of data is used here that might have highly affected the results quality')

### App main pic
image1 = Image.open('peanutbutter pie.png')
st.image(image1)


#######################################################################################################################################
### DATA LOADING

### A. define function to load data
@st.cache_data # <- add decorators after tried running the load multiple times
def load_data(path):

    df = pd.read_parquet(path)

    return df

### B. Load 
df = load_data('Transformed Data/ Cleaned-Sampled-Recipes_with_images_SampledForAPP.parquet')


### C. Display the dataframe in the app
#st.dataframe(df)

#######################################################################################################################################
### Content-Base Recommended system
st.header("Recommended by Recipe Name")
st.write("This recommendation model is based on recipes' description, keywords, and ingredients similarity - Content Based Model")
#######################################################################################################################################
### Give some options to choose 

st.subheader("Available Recipes")
st.write("Here are a sample of available recipes. Choose one you like more to start trying the app.")
images=['https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/v1/img/recipes/33/00/9/aq388traRNtzvikCi3vA-IMG_2527.JPG',
        'https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/v1/img/recipes/29/66/92/x4otBVBETYeIw89qbkJg_chicken%20marsala%20SITE-3.jpg',
        'https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/v1/img/recipes/29/26/07/picaA4uXw.jpg',
        'https://img.sndimg.com/food/image/upload/w_555,h_416,c_fit,fl_progressive,q_95/v1/img/recipes/67/06/8/pic2hWutw.jpg'
        ]
captions=['Peanut Butter Pie','Skillet Chicken Marsala','Chinese Beef With Broccoli','Chewy Delicious Chocolate Chip Cookies']
st.image(images, caption=captions,width=170)



### get the input
text = st.text_input('Enter your favorite recipe name below:', 'Chinese Beef With Broccoli')

#######################################################################################################################################
### Finding the output
def content_recommender_second(recipename, df, similarities, RevCount_threshold=10) :
    
    # Get the recipe by the name
    Recipe_index = df[df['Name'] == recipename].index
    
    # Create a dataframe with info of recipe
    sim_df = pd.DataFrame(
        {'RecipeName': df['Name'],'RecipeId':df['RecipeId'],'Description': df['Description'],'rating':df['AggregatedRating'],'rev':df['ReviewCount'],
         'category':df['RecipeCategory'],'cal':df['Calories'],'Keywords':df['Keywords'],'ingredients':df['RecipeIngredientParts'],
         'Images':df['Images'],'similarity': np.array(similarities_second[Recipe_index, :].todense()).squeeze()
        })
    
    # Get the top 10 recipe with > 10 review
    top_recipe = sim_df[sim_df['rev'] > RevCount_threshold].sort_values(by='similarity', ascending=False).head(10)
    
    #drop the input recipe
    top_recipe = top_recipe[top_recipe['RecipeName'] != recipename]
    
    return top_recipe


similarities_second = joblib.load('similarities_second_image_smalldate.pkl')
similar_recipe = content_recommender_second(text, df, similarities_second, RevCount_threshold=4)
similar_recipe.reset_index(inplace=True)

#######################################################################################################################################
### SHOW OUT PUTS
st.subheader("You also may like these recipes:")
top5recipe = similar_recipe.head()

Image_list=[]
caption=[]
for row in top5recipe['Images']:
    Image_list.append(row[0])

for row in top5recipe['RecipeName']:
    caption.append(row)

st.image(Image_list,width=130,caption=caption)

#######################################################################################################################################
st.title("FUNK SVD")
st.write("Recommending to current users based on what they and other users have rated.")
sorted_df = load_data("Transformed Data/ FunksvdReviewdf.parquet")
recipe_df = load_data("Raw data/ recipes_twocolumn.parquet")
#######################################################################################################################################
## FunkSVD Function
def recom_Rating_FunkSVD(df,user_id):
    
    #Surprise format data
    my_dataset = Dataset.load_from_df(sorted_df, Reader(rating_scale=(1, 5)))
    my_train_dataset = my_dataset.build_full_trainset()
    
    my_algorithm = FunkSVD(n_factors=10,n_epochs=100, 
    lr_all=0.1,    # Learning rate for each epoch
    biased=False,  # This forces the algorithm to store all latent information in the matrices
    verbose=0,random_state=10)
    
    my_algorithm.fit(my_train_dataset)
    
    U = my_algorithm.pu
    Re = my_algorithm.qi.T

    
    # find the inner representation of the user
    inner_user_id = my_train_dataset.to_inner_uid(user_id)  
    user_profile = U[inner_user_id]
    
    #ALL RECIPES
    recipe_profile = Re[:, :]
    
    #predict rating
    expected_rating = np.dot(user_profile, recipe_profile)
    
    #match ratings with recipeids
    predicted_df = pd.DataFrame({'recipeid':df['RecipeId'].unique(),'predicted_rating':expected_rating})
    predicted_df = predicted_df[~(predicted_df['recipeid'].isin(rated_recipes_info['RecipeId'].values))]
    
    #top rated recipes IDs
    top_recipes_id = predicted_df.sort_values('predicted_rating',ascending=False).head()['recipeid'].values
    
    #Find recipe names
    out_put = recipe_df[recipe_df['RecipeId'].isin(top_recipes_id)]['Name']
    
    return out_put
    
#######################################################################################################################################
### get the input
user_id = st.number_input('Enter a userid:', min_value= None,value=1064617)
st.write("Sample of available userids: 420929, 235355, 567976, 589581 ")

## what they rated
rated_recipe = sorted_df[sorted_df['Review_AuthorId']==user_id]
recipe_info = recipe_df[recipe_df['RecipeId'].isin(rated_recipe['RecipeId'].values)]
merged = pd.merge(rated_recipe,recipe_info, left_on='RecipeId', right_on='RecipeId')
rated_recipes_info = merged[['Name','Review_Rating','RecipeId']].sort_values('Review_Rating',ascending=False)    
st.write('This user rated:')
st.dataframe(rated_recipes_info[['Name','Review_Rating']])

## recommend
st.write('Our recommendation:')
output_df = recom_Rating_FunkSVD(sorted_df,user_id)
st.dataframe(output_df)




### 
st.write("-----------------------------------------------------------------------------------------------------------------------------")
st.write('For backend code, analysis and more detail, please visit my github: github.com/fyazdi75/Recommender-System-for-Food-Recipes')
st.write('Contact me to have a coffee or cook one of recipes :) ')
st.write('Faezeh Yazdi')
st.write('July 2023')
