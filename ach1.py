import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
model = joblib.load('/home/chaitanya/ach/model.joblib')
vectorizer = joblib.load('/home/chaitanya/ach/vectorizer.joblib')

# Load the dataset
df = pd.read_csv('/home/chaitanya/ach/tourism.csv')
# Enable wide mode
st.set_page_config(layout="wide")

def predict_location_name(Region_input,):
    new_ingredients = [ingredient.strip() for ingredient in ingredients_input.split(',')]
    new_ingredients_vectorized = vectorizer.transform(new_ingredients)
    predicted_recipe_names = model.predict(new_ingredients_vectorized)
    return predicted_recipe_names


def get_recipe_info(recipe_name):
    recipe_info = df[df['TranslatedRecipeName'] == recipe_name]
    return recipe_info.iloc[0]


def main():
    st.title('Recipe Generator')
    st.write('Enter Ingredients Available:')

    # Collect new ingredients from the user in a single text input field
    ingredients_input = st.text_input('Ingredients', '')

    # Make predictions when the user clicks the "Predict" button
    if st.button('Generate'):
        predicted_recipe_names = predict_recipe_names(ingredients_input)

        st.header('Here are few recipes that you would like:')
        for recipe_name in predicted_recipe_names:
            recipe_info = get_recipe_info(recipe_name)  # Move this line here

            # Create two columns to display information (one for text, one for image)
            col1, col2 = st.columns([3, 1])

            # Column 1 (information except the image - left)
            with (col1):
                st.subheader(recipe_name)
                st.write('URL:', recipe_info['URL'])
                st.subheader('Additional Information:')
                st.write(f'Total Time (in mins): {recipe_info["TotalTimeInMins"]}')
                st.write(f'Number of Ingredients: {recipe_info["Ingredient-count"]}')
                st.write('Ingredients:', recipe_info['Cleaned-Ingredients'])

                # Split the translated instructions at each full stop
                translated_instructions = recipe_info['TranslatedInstructions']
                instructions_list = translated_instructions.split('. ')

                # Display the ordered list of instructions
                st.subheader("Ouick Instructions:")
                st.markdown("<ol>" + "<br>".join("<li>" + instr + "</li>" for instr in instructions_list) + "</ol>",
                            unsafe_allow_html=True)

            # Column 2 (image - right)
            with col2:
                st.image(recipe_info['image-url'], use_column_width=True)

            st.write('-' * 50)


if _name_ == '_main_':
    main()