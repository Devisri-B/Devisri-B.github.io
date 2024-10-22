import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import spacy

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load CSV data
data = pd.read_csv('new_data.csv')

# Load the fine-tuned GPT-2 model
model_name = './fine_tuned_meal_plan_model'  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Spoonacular API key
spoonacular_api_key = '3f8c82acd48547a49f712a2ab1cdf9b1'  

# Dictionary to hold user-selected recipe state and preferences
user_state = {}

# Function to generate a response using the fine-tuned GPT-2 model
def generate_response(user_input):
    print(f"Generating response for input: {user_input}")
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=250, do_sample=True, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"Generated response: {response}")
    return response


# Load the SpaCy language model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Function to handle plural and singular forms using SpaCy
def handle_plural_ingredient(ingredient):
    # Use SpaCy to lemmatize the word (convert plural to singular)
    doc = nlp(ingredient)
    lemmatized = [token.lemma_ for token in doc][0]  # Get the lemmatized form

    # If the ingredient is already plural, add both singular and plural forms
    if ingredient != lemmatized:
        return [ingredient, lemmatized]    # Plural form comes first, singular form second

    # Otherwise, if the ingredient is singular, return both singular and regular plural form
    else:
        if ingredient.endswith('s'):
            plural_form = ingredient  # If it already ends in 's', treat it as plural
        elif ingredient.endswith('y'):
            plural_form = ingredient[:-1] + 'ies'  # Handle nouns like 'berry' -> 'berries'
        else:
            plural_form = ingredient + 's'  # Regular plural form

        return [plural_form, lemmatized ]   # Plural form comes first, singular form second



# Function to filter meal plans based on cooking time, cuisine, ingredients, and recipe name
def get_meal_plan(max_time_in_minutes=None, cuisine=None, ingredients=None, recipe_name=None):
    filtered_recipes = data

    # Filter by cooking time (for general queries)
    if max_time_in_minutes:
        filtered_recipes = filtered_recipes[filtered_recipes['TotalTimeInMins'] <= max_time_in_minutes]

    # Filter by cuisine (for general queries)
    if cuisine:
        filtered_recipes = filtered_recipes[filtered_recipes['Cuisine'].str.contains(cuisine, case=False)]

    # Filter by ingredients (for general queries)
    if ingredients:
        for ingredient in ingredients:
            # Get both singular and plural forms of the ingredient
            ingredient_variants = handle_plural_ingredient(ingredient)
            print(f"Ingredient variants: {ingredient_variants}")

            # Create a regex pattern to match both singular and plural forms
            ingredient_pattern = '|'.join(ingredient_variants) 
            print(f"Ingredient pattern: {ingredient_pattern}")

            # Filter recipes that contain the current ingredient in 'Cleaned-Ingredients'
            filtered_recipes = filtered_recipes[filtered_recipes['Cleaned-Ingredients'].str.contains(ingredient_pattern, case=False)]
            
            # Print remaining recipes count after filtering by this ingredient
            print(f"Remaining recipes after filtering by {ingredient}: {len(filtered_recipes)}")

            # If no recipes match, stop filtering further
            if filtered_recipes.empty:
                print(f"No recipes found with the ingredient: {ingredient}")
                break

    # Filter by recipe name (for specific recipe name queries)
    if recipe_name:
        filtered_recipes = filtered_recipes[filtered_recipes['TranslatedRecipeName'].str.contains(recipe_name, case=False)]

    # If the query is ingredient or time-based, return up to 10 random results
    if not filtered_recipes.empty:
        meal_plan = filtered_recipes[['TranslatedRecipeName', 'TotalTimeInMins', 'URL', 'TranslatedIngredients', 'image-url', 'Cleaned-Ingredients', 'TranslatedInstructions']].sample(n=min(10, len(filtered_recipes))).to_dict(orient='records')
    else:
        meal_plan = []

    return meal_plan

# Function to interpret user query and extract relevant information
def parse_user_query(user_message):
    # Extract time (in minutes) from the user message
    time_match = re.search(r'(\d+)\s*(?:min(?:utes?)?)?', user_message.lower())
    max_time = int(time_match.group(1)) if time_match else None
    print(f"Extracted time: {max_time}")

    # Extract potential cuisine
    cuisine_match = re.search(r'(indian|south indian|andhra|chinese|italian|mexican)', user_message.lower())
    cuisine = cuisine_match.group(1) if cuisine_match else None
    print(f"Extracted cuisine: {cuisine}")

    # Extract potential ingredients (supports both singular and plural)
         # Using regex to find ingredients mentioned after "with" or "and"
    ingredient_match = re.findall(r'with|has ([\w\s,]+)', user_message.lower())
    ingredient_list = []
    if ingredient_match:
        # Split ingredients by commas and 'and'
        for ingredient_str in ingredient_match:
            ingredients = re.split(r',|\sand\s', ingredient_str)
            ingredient_list.extend([ingredient.strip() for ingredient in ingredients if ingredient.strip()])
    print(f"Extracted ingredients: {ingredient_list}")
    
    
    # Extract recipe name, excluding common words
    excluded_words = ['a', 'the', 'any', 'want', 'need', 'like', 'give']
    # Search for the word before "recipe"
    recipe_name = None  # Default value in case no recipe name is found
    recipe_match = re.search(r'\b(\w+)\s+recipe', user_message.lower())
    if recipe_match:
        # Extract the word before "recipe"
        potential_recipe_name = recipe_match.group(1)
        # Check if the extracted word is in the excluded words list
        if potential_recipe_name not in excluded_words:
            recipe_name = potential_recipe_name
    print(f"Extracted recipe name: {recipe_name}")
    
    return max_time, cuisine, ingredient_list, recipe_name

def process_user_request(user_message):
    # Handle casual conversation
    if 'hello' in user_message.lower():
        return "Hello! How can I assist you today?"
    if 'how are you' in user_message.lower():
        return "I'm doing great! How about you?"

    # Parse the user's message to extract time, cuisine, ingredient, and recipe name preferences
    max_time, cuisine, ingredients, recipe_name = parse_user_query(user_message)

    # If no ingredients, recipe name, or other relevant info is found, return None to fall back to Spoonacular
    if not ingredients and not recipe_name and not cuisine and not max_time:
        print("No relevant data found in user input. Falling back to Spoonacular API.")
        return None

    # If the user is searching for a specific recipe by name
    if recipe_name:
        meal_plan = get_meal_plan(recipe_name=recipe_name)
        if meal_plan:
            selected_recipe = meal_plan[0]  # Return the first matching recipe (assumed unique)
            response = f"<strong>Here is the recipe for </strong>{selected_recipe['TranslatedRecipeName']}<br>\n"
            response += f"<strong>Ingredients:</strong> {selected_recipe['TranslatedIngredients']}<br>\n"
            response += f"<strong>Procedure:</strong> {selected_recipe['TranslatedInstructions']}<br>\n"
            response += f"<strong>Need more info?:</strong> <a href='{selected_recipe['URL']}' target='_blank'>{selected_recipe['URL']}</a><br>\n"
            response += f"<img src='{selected_recipe['image-url']}' alt='Recipe Image' width='200'><br>"
            return response
        else:
            # If no recipe is found, fall back to Spoonacular API
            return None

    # Handle time, cuisine, or ingredient-based queries (meal suggestions)
    else:
        meal_plan = get_meal_plan(max_time_in_minutes=max_time, cuisine=cuisine, ingredients=ingredients)
        if meal_plan:
            user_state['meal_plan'] = meal_plan
            # Display a list of up to 10 random meal options
            response = "Here are some meal options:\n"
            for idx, recipe in enumerate(meal_plan):
                response += f"<button>{idx + 1}. {recipe['TranslatedRecipeName']} (Takes {recipe['TotalTimeInMins']} minutes)</button><br>"
            response += "\nPlease select a recipe by clicking a button or let me know if you prefer a different cuisine or ingredient."
            return response
        else:
            # If no recipes found in the local data (CSV), return None to fall back to Spoonacular
            return None


# Handle selected recipe
def handle_recipe_selection(user_message):
    if user_message.isdigit() and 'meal_plan' in user_state:
        selected_idx = int(user_message) - 1
        if 0 <= selected_idx < len(user_state['meal_plan']):
            selected_recipe = user_state['meal_plan'][selected_idx]
            # Display the selected recipe's ingredients, instructions, URL, and image
            response = f"<strong>Ok, good choice! Here's the recipe for </strong>{selected_recipe['TranslatedRecipeName']}<br>\n"
            response += f"<strong>Ingredients: </strong>{selected_recipe['TranslatedIngredients']}<br>\n"
            response += f"<strong>Procedure: </strong>{selected_recipe['TranslatedInstructions']}<br>\n"
            response += f"<strong>Need more info?: </strong><a href='{selected_recipe['URL']}' target='_blank'>{selected_recipe['URL']}</a><br>\n"
            response += f"<img src='{selected_recipe['image-url']}' alt='Recipe Image' width='200'><br>"
            return response
        else:
            return "Invalid selection. Please select a valid recipe number."
    else:
        return None
    
# Function to fetch from Spoonacular if no data is found in CSV or GPT
def fetch_from_spoonacular(query):
    print(f"Fetching recipe from Spoonacular API for query: {query}")
    url = f"https://api.spoonacular.com/recipes/complexSearch?query={query}&apiKey={spoonacular_api_key}&addRecipeInformation=true&fillIngredients=true"
    response = requests.get(url)
    
    # Check if the response was successful (status code 200)
    if response.status_code != 200:
        print(f"Error from Spoonacular API: {response.status_code}, {response.text}")
        return {"error": f"Failed to fetch recipe. Error: {response.status_code}"}
    
    data = response.json()
    
    # Safely check if the 'results' key exists
    if 'results' in data and len(data['results']) > 0:
        recipe = data['results'][0]
        response = {
            "recipe_name": recipe['title'],
            "ingredients": ', '.join([ingredient['name'] for ingredient in recipe.get('extendedIngredients', [])]),
            "instructions": recipe.get('instructions', 'Instructions not available'),
            "url": recipe.get('sourceUrl', 'URL not available'),
            "image_url": recipe.get('image', 'Image not available')
        }
        print(f"Found recipe from Spoonacular: {response}")
        return response
    else:
        print("No results found in Spoonacular API response.")
        return {"error": "No recipes found for this query."}

# Helper function to format the response for display
def format_response(response_data):
    return (
        f"<strong>Here is the recipe for </strong>{response_data['recipe_name']}<br>\n"
        f"<strong>Ingredients:</strong> {response_data['ingredients']}<br>\n"
        f"<strong>Procedure:</strong> {response_data['instructions']}<br>\n"
        f"<strong>Need more info?:</strong> <a href='{response_data['url']}' target='_blank'>{response_data['url']}</a><br>\n"
        f"<img src='{response_data['image_url']}' alt='Recipe Image' width='200'><br>"
    )

# Flask route for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    print(f"User input: {user_input}")

    # Step 1: Search the data.csv file
    # Check if the input starts with a digit (recipe selection)
    # Extract the first digit which corresponds to the selected recipe number
    recipe_selection_match = re.match(r'(\d+)', user_input)
    
    if recipe_selection_match:
        # Extract the recipe number from the input
        selected_recipe_number = recipe_selection_match.group(1)
        print(f"Selected recipe number: {selected_recipe_number}")
        
        # Handle recipe selection by the user
        bot_reply = handle_recipe_selection(selected_recipe_number)
        return jsonify({'response': bot_reply})
    
    
    # If it's not a digit, process the user query to display the meal plan or recipe
    bot_reply = process_user_request(user_input)
    if bot_reply:
        return jsonify({'response': bot_reply})
    
    # Step 2: If no relevant data found in CSV, fallback to Spoonacular API
    print("Querying Spoonacular API for external recipe.")
    external_response = fetch_from_spoonacular(user_input)
    if 'error' not in external_response:
        return jsonify({'response': format_response(external_response)})
    else:
        #Return an error message if no results found in Spoonacular either
        return jsonify({'response': external_response['error']})
  
    #find data from fine_tuned model
    # fine_tuned_response = generate_response(user_input)
    # if fine_tuned_response:
    #     return jsonify({'response': fine_tuned_response})

if __name__ == '__main__':
    app.run(debug=True)
