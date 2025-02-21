# Programming Assignment 7: Chatbot! 

**Late days CANNOT be used on this assignment. Please submit early and often to avoid last minute submission issues!**

**You must work in groups of 3-4 members. (To work in a group of size 2, you must get special permission from the staff.) All submissions will be graded according to the same criteria, regardless of group size.**

In this assignment you will build a chatbot somewhat like the chatbot ELIZA that you saw in the lecture. You will work on a simpler, single-agent version first, and then choose from a set of advanced features to get more points. **A score of 85 points for the coding portion is considered full credit (1 rubric + 60 Starter (GUS) + 8 LLM Prompting + 16 LLM Programming).  For LLM Prompting and LLM Programming mode you will choose which of the parts to implement.  You must implement the full starter mode.  The full assignment (coding + written) is worth 100 points.**

To get you started, we will guide you in the implementation of a chatbot that can recommend movies to the user – this will be referred to as the **starter** mode. After you have implemented the movie-recommender chatbot, we will try to replicate the bot you wrote by prompting a Large Language Model (LLM) by running your bot in **llm prompting** mode.  Finally you will switch to an **llm programming** mode which allows you to use both your own code and LLM calls to implement an interesting extension of the movie recommender. Be creative, and have fun!

PA7 has 3 Submission Components on Gradescope.  Make sure you complete them all!

 - PA7 Coding (85 Points)
 - PA7 User Testing (5 Points)
 - PA7 Reflection + Ethics Questions (10 points)

 ## Important Setup Note

Although this assignment mostly reuses the environment you set up in PA0, we need one additional package.  Recall to activate your PA0 environment you will use:

    conda activate cs124

Please run this command after activating your cs124 environment.

    pip install openai

## Contents of this file!
This is a big file! Here are the main sections:

**Background and Overview:**
- [Background](#a-little-bit-of-background)
- [High Level Spec](#high-level-specification)

**(Part 1) Getting started: Starter Mode (GUS), REPL, and resources**
- [Starting to code: warm up](#starting-to-code-warm-up)
- [Starter code and REPL](#running-the-starter-code)
- [Starter Mode Overview](#part-1-starter-mode---gus-60-points)
- [Writing code for starter mode](#writing-code-for-starter-mode)
- [Movie Database](#your-movie-database-movielens)
- [Sentiment Lexicon](#your-sentiment-lexicon)

**Debugging and Testing**
- [Running sanity check](#sanity-check)
- [Local script testing](#local-script-testing)
- [Printing debug info](#printing-debug-info) We give you some built in debugging tools! Check them out!
- [Important tips and notes](#important-notes-and-tips) **Important!!**
- [Other resources and code size](#other-resources-and-code-size)

**(Part 2) LLM Prompting Mode**
- [LLM Prompting Mode Overview](#part-2-llm-prompting-mode-8-points)
- [LLM Prompting Mode Rubric](#llm-prompting-mode-rubric)
- [Running in LLM Prompting Mode](#running-in-llm-prompting-mode)

**(Part 3) LLM Programming Mode**
- [LLM Programming Mode Overview](#part-3-llm-programming-mode-16-points)
- [LLM Programming Mode Rubric](#llm-programming-rubric)
- [Switching to LLM Programming Mode](#running-in-llm-programming-mode)
- [How are starter and LLM Programming mode different?](#difference-between-starter-mode-and-llm-programming-mode)

**Evaluation and Rubric (answers to your grading questions!)**
- [Evaluation](#evaluation) (the rubric is here!)

**Submitting**
- [Submitting Your Solution](#submitting-your-solution) As always, submit **early AND often!!**

**Handy Links**
- You can find links to [all relevant resources](#links-to-resources) here so that you don't miss anything!

## A little bit of background
In the early 1960s, the well-known early artificial intelligence (AI) program ELIZA made a big impact after showing its ability to process input and produce output expressed in natural language. [ELIZA](https://en.wikipedia.org/wiki/ELIZA) was one of the first conversational agents in AI, designed to simulate intelligent conversations, also known as chatbots. Developed by [Joseph Weizenbaum](https://en.wikipedia.org/wiki/Joseph_Weizenbaum) at MIT, ELIZA had several modules to interact in English, German, and Welsh, and different modes of operation. The most famous was a module that simulated the interaction with a [Rogerian psychotherapist](https://en.wikipedia.org/wiki/Person-centered_therapy). A typical conversation with ELIZA (all caps) was as follows:

    You are like my father in some ways.
    WHAT RESEMBLANCE DO YOU SEE
    You are not very aggressive but I think you don't want me to notice that.
    WHAT MAKES YOU THINK I AM NOT AGGRESSIVE
    You don't argue with me.
    WHY DO YOU THINK I DON'T ARGUE WITH YOU
    You are afraid of me.
    DOES IT PLEASE YOU TO BELIEVE I AM AFRAID OF YOU`

After the initial success by ELIZA, a number of other chatbots were created with increased complexity – as an example, on the late 70s a chatbot named [PARRY](https://en.wikipedia.org/wiki/PARRY), created by Stanford’s Psychiatry professor [Kenneth Colby](https://en.wikipedia.org/wiki/Kenneth_Colby), simulated a psychotic patient that used a belief system internally, allowing for more complex behavior.

You will implement a chatbot similar to ELIZA, but even more sophisticated!  Using frame-based dialogue we will implement a dialogue agent using similar ideas to the GUS agent described in lecture! This chatbot will recommend movies to the user. Let’s get started!

## High-level specification
Conversational agents like ELIZA or GUS can be decomposed into four modules (see Jurafsky and Martin 2008, [Norvig 1991](https://github.com/norvig/paip-lisp)), each one responsible for a specific task:

1. Read the input.
2. Extract relevant information from the input, which can be domain specific – as in our movie recommender chatbot.
3. Transform the input into a response – users normally expect a response in the given domain.
4. Print the response.

The starter code provided for the assignment handles modules (1) and (4) for you. You will be responsible for implementing modules (2) and (3), in both the starter (GUS) and LLM Programming modes.

Once you have set up the environment, the warm-up section will help you familiarize yourself with the REPL. After you feel comfortable interacting with the REPL, you will implement the extraction and transformation modules for the movie recommender chatbot.

## Starting to code: warm up
Modules (1) and (4) are provided to you in this assignment using a common pattern in software engineering known as the Read-Eval-Print-Loop, REPL. The REPL creates a prompt that gets an input string from the user (module 1). Then the REPL passes the input to a chatbot class that is responsible for doing the work in modules (2) and (3). The response generated by the chatbot class is then handled again by the REPL, which prints the response and waits for more input. Let’s see how it works!

## Running the starter code
In the starter code folder, fire up the REPL by issuing the following command:
`python3 repl.py`

## Interacting with the REPL
You can type your message in the prompt to moviebot, our default chatbot, and hit enter. To exit the REPL, write `:quit` (or press Ctrl-C to force the exit). Right now the chatbot is not doing anything more than printing the string you entered – concatenated to a string that says it processed it.

## Interacting with REPL via testing scripts
In the testing/test_scripts/ folder, you will find some testing scripts. To run a script, enter the following command (replace testing/test_scripts/simple.txt with your desired script):

    python3 repl.py < testing/test_scripts/simple.txt
    
*[For Windows users: If you are running the above in PowerShell and see an error, try running in cmd instead!]*

As you can see, each line of `simple.txt` is entered consecutively into the chatbot. However, the script terminates once it hits `:quit` and any lines afterwards will not be executed.

This will be useful when you want to test the same script multiple times. We will be testing your code with similar (but not the same) scripts as the ones provided in test_inputs.

## Customizing your chatbot
All the code that you will need to write for this assignment will be in `chatbot.py`. Open that file and follow the instructions there to change the name of the chatbot, the greeting message, and the goodbye message. You can change your chatbot’s name now!

# (Part 1) Starter Mode - GUS (60 points)
In starter mode, your chatbot will help the user by giving movie recommendations. It will ask the user to say something about movies they have liked, and it will come up with a recommendation based on those data points. The user of the chatbot in starter mode will follow its instructions, so you will deal with fairly restricted input. Specifically, you can assume:
- **Movie names** will come in **quotation marks**
- **Expressions of sentiment** will be **simple**

Here is an example of how this might go:

    moviebot> Hi! I'm MovieBot! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen.
    > I really liked Moneyball.

The bot gets confused because it expects quotation marks.

    moviebot> Sorry, I don't understand. Tell me about a movie that you have seen.
    > I really liked "Moneyball".
    moviebot> You liked "Moneyball". Thank you! Tell me about another movie you have seen.
The bot understands everything, so it echoes your sentiment. It’s a bit repetitive, though.

    > I was completely amazed at how "Rush Hour 3" was pretty much the most ridiculous thing I've seen in my entire life.
This sentence was too hard for the starter bot. That’s okay!

    moviebot> I'm sorry, I'm not quite sure if you liked "Rush Hour 3". 
    Tell me more about "Rush Hour 3".
    > I hated "Rush Hour 3".
    moviebot> You did not like "Rush Hour 3". Thank you! 
    Tell me about another movie you have seen.
    > Oh, I also thought "The Little Mermaid" was fantastic!
    moviebot> You liked "The Little Mermaid". Thank you! Tell me about another movie you have seen.
    > I'm tired of typing about movies.    
The bot gives a nice response to this! It’s okay for the chatbot to require 5 data points before giving a recommendation:

    moviebot> I want to hear more about movies! Tell me about another movie you have seen.
    I didn't like "The Notebook" or "Avatar".
The bot can’t handle this, but that’s okay.

    moviebot> Please tell me about one movie at a time. Go ahead.
    I didn't like "The Notebook".
The bot makes a mistake in sentiment extraction in its reply below. This was a pretty straightfoward sentence, so this mistake should be penalized.

    moviebot> You liked "The Notebook". Thank you! 
    That's enough for me to make a recommendation.
    I suggest you watch "In the Heart of the Sea". 
    Would you like to hear another recommendation? (Or enter :quit if you're done.)
    > :quit
    moviebot> Thank you for hanging out with me! Stay in touch! Goodbye!
It’s up to you to provide appropriate guidance for the user, so that they give you the output you expect, and to make your chatbot as natural and as engaging a speaker as you can. The following section gives you a high-level perspective of a possible API.

# Writing code for starter mode
The REPL class calls the process method in the Chatbot class each time the user enters a string. Your implementation of process should handle the input, update internal data structures accordingly, and output a nice response for the user.

### Implementing core logic: `extract_titles`, `find_movies_by_title`, `extract_sentiment`, and `recommend`
We've decomposed some of the core logic for you. You will need to implement the following four methods, per the specifications [in the rubric](#evaluation):

`extract_titles(preprocessed_input)`: given an input text, output a list of plausible movie titles that are mentioned in text, i.e. substrings for the bot to look up in the movie database. For starter mode, this function should simply output all substrings that were found between quotation marks.

`find_movies_by_title(title)`: return a list of indices corresponding to titles in the movie database matching the given title. As a baseline, you are expected to handle:
- Titles with or without the year included
    - If a year is not provided, a title may match multiple
    - find_movies_by_title("Titanic") should return [1359, 2716], corresponding to the 1997 and 1953 versions
    - find_movies_by_title("Titanic (1997)") should return simply [1359]
- The way many movies in the database have English articles (a, an, the) moved to the end. 
    - For example, find_movies_by_title("The American President") should return [10], where 10 is the index of "American President, The (1995)". 

`extract_sentiment(preprocessed_input)`: extract the sentiment of text. For starter mode:
- return -1 if text has negative sentiment
- 1 if positive sentiment
- 0 if no non-neutral sentiment detected.
- If you're not sure where to get started, check out [the provided lexicon](#your-sentiment-lexicon) and think about how you might use this with techniques learned on previous PAs!

`recommend(user_ratings, ratings, k)`: This function should:
- Input the **provided vector of the user's preferences** and a **pre-processed matrix of ratings by other users** (See the following section for details on pre-processing)
- Use collaborative filtering to **compute and return** a list of the **k movie indices** with the highest recommendation score that the user hasn't seen. 
- For starter mode, you will implement and use item-item collaborative filtering, with **cosine similarity** and **NO rating normalization**. For the neighbor selection step, you can assume that all items rated by the user are used as neighbors (the final prediction is a weighted average of ratings on all interacted items).

`binarize(ratings, threshold)`: Binarizes the ratings matrix (for more details, see the [movie database section](#your-movie-database-movielens)). This function should:
- Replace all entries **above the threshold** with **1**
- Replace all entries **below or equal to the threshold** with **-1**
- Leave entries that are 0 as 0
- Be called at the **beginning of init** 
    - You will be working with a binarized ratings matrix for the rest of the assignment!
- `threshold` is a parameter passed to `binarize`, set to 2.5 by default

These functions will be tested by the autograder, so make sure you implement them correctly! The starter code includes a sanity check script that runs some basic tests on these functions to help you debug them. Once you've debugged locally, we **highly recommend** you submit to Gradescope to test these with the autograder. 

We advise that you explore these and other function stubs provided in the starter code and call them in process as appropriate.

### Combining your functions in `process(line)`
Once you have implemented the above functions, you can integrate them into `process(line)`, which crafts a response for a given input. 

Unlike the autograder, your users won't be able to read the output of the helper functions above to tell if you extracted the right sentiment and movie title. To give them a chance to correct you, your bot is **required** to echo the sentiment and the movie title that it extracts from the user’s input. You can do this creatively; see the example above for one possibility. **If you don’t echo this information, you cannot get maximum points for title extraction and sentiment extraction.**

There are many different ways to implement modules (2) (extracting relevant information from the input) and (3) (crafting relevant information into a response) for the chatbot. Read the [rubric](https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing) carefully to see the functionality that will be evaluated, and implement that first! The next two sections give you information on the data provided in the starter code.

# Your movie database: MovieLens
The data comes from [MovieLens](https://movielens.org/) and consists of a total of 9125 movies rated by 671 users The starter code provides a 9125 x 671 utility matrix that contains ratings for users and movies. The ratings range anywhere from 1.0 to 5.0 with increments of 0.5. For starter mode, you will implement binarize method and use it to binarize the ratings as follows:

    +1 if the user liked the movie (3.0-5.0)
    -1 if the user didn’t like the movie (0.5-2.5)
    0 if the user did not rate the movie
The starter code also provides a list with 9125 movie titles and their associated movie genres. The list is formatted as expected: the movie in the first position in the list corresponds to the movie in the first row of the matrix. An example entry looks like:

    ['Blade Runner (1982)', 'Action|Sci-Fi|Thriller']
As in previous homeworks, you can inspect the data to have an idea of the variations in how the titles are represented in this dataset. Note that the second element in the list/array is a string with movie genres separated by the pipe symbol |.

## Your sentiment lexicon
The starter code also provides a sentiment lexicon that you can use to extract sentiment from the input. It consists of 3624 words with their associated sentiment extracted from [Harvard Inquirer](https://inquirer.sites.fas.harvard.edu/spreadsheet_guide.htm) (Stone et al. 1966). The lexicon is stored for your convenience in a dictionary/hash map, where the word is the key and the sentiment the value.  You'll find the lexicon in the `/data` folder.

## Sanity check
We have provided a sanity check script to test your code on basic inputs. To run the starter mode sanity checks, run `python3 testing/sanitycheck.py`
### Local script testing
As mentioned before, you can run individual test scripts located in the testing/test_scripts/ directory with the following command:

    python3 repl.py < testing/test_scripts/simple.txt

*[For Windows users: If you are running the above in PowerShell and see an error, try running in cmd instead!]*

You can also run all the scripts with the following commands:

    sh testing/run_all_scripts.sh
You should check the outputted transcripts to make sure that your chatbot doesn't crash or fail incoherently. We will not be grading on these exact scripts (nor do the given test scripts cover all the requirements given in the rubric), but the method of running scripts will be used. Feel free to add your own test scripts as well! If you do, don't forget to add :quit as your last line or else the script will never terminate.

**Note:** passing locally does **NOT** guarantee that you will pass the autograder, so please submit **early and often**!!

# Printing debug info
The debug method in the Chatbot class can be used to print out debug information about the internal state of the chatbot that you may consider relevant, in a way that is easy to understand. This is helpful while developing the assignment, but will not be graded. To enable debug info, type in the chatbot session in the REPL

    :debug on

and type

    :debug off

to disable it.

# Important Notes and Tips
To ensure your recommend function's output matches the autograder's, make sure to implement collaborative filtering as described in the comments! As a reminder: item-item collaborative filtering, cosine similarity, no rating normalization, and all rated items used as neighbors.

We encourage you to use numpy for this assignment. [np.dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html), [np.linalg.norm](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html), and [np.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html) may be especially helpful.

If you encounter any issue, we strongly encourage you to read the [FAQs](https://docs.google.com/document/d/1y4iy7pcS3S4K9ChBSQ21Tjj-HyEKmvn7bytM-RTMwoI/edit?usp=sharing) doc first, because it may give you answer faster than the CAs!

# Connecting to Together API

For LLM Prompting Mode and LLM Programming Mode we need external LLM API access.  Together AI has graciously granted us free credits to use for this assignment.  We provide a brief guide on getting an API key and linking it this assignment [here](https://docs.google.com/document/d/1N5chC5b15ls-XXcpfjhSx71854fmvb_4DGD4qxkT0LU/edit?usp=sharing).

# (Part 2) LLM Prompting Mode (8 points)

LLMs are a powerful tool that drive a lot of modern chatbots!  Let's get some experience building an LLM chatbot.  For this part of the assignment we will use the [Mixtral 8x7B Model](https://huggingface.co/docs/transformers/en/model_doc/mixtral).  Mixtral is a blend of eight 7-billion parameter language models!

This part of the assignment requires only modifying one function `def llm_system_prompt(self)` in chatbot.py.

The system prompt of an LLM serves as a prefix to the entire conversation and is often used to describe the role that the LLM will take on for all remaining turns of the conversation.

We provide this default system prompt for you:

```python
system_prompt = """Your name is moviebot. You are a movie recommender chatbot. You can help users find movies they like and provide information about movies."""
```

This works to get the chatbot to provide recommendations, but it does a whole lot of other things too:

```
moviebot> How can I help you?
> What is AutoTrader?
moviebot>  AutoTrader is a popular online marketplace for buying and selling new and used cars. It allows users to search for cars based on various criteria such as make, model, price, mileage, and location. AutoTrader also provides tools and resources to help users make informed decisions about buying or selling a car, such as pricing guides, vehicle history reports, and financing options.

AutoTrader was founded in 1997 and is based in Atlanta, Georgia. It operates in the United States, United Kingdom, and Canada. In addition to its website, AutoTrader also has mobile apps for iOS and Android devices.

AutoTrader's mission is to make the car buying and selling process easy and transparent for consumers. It offers a wide selection of cars from dealerships and private sellers, and provides a platform for users to connect and negotiate deals. AutoTrader also offers advertising and marketing services for dealerships and automotive manufacturers.
```
Uh oh!  We want a movie recommender bot, not a car sales expert!  Let's get this bot under control.  

## LLM Prompting Mode Rubric

By just modifying the system prompt, choose to implement the features outlined in [the rubric](https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing) to get up to 8 points.

How do we do this by just modifying the system prompt?  This is a skill called prompt engineering!  If you're looking for inspiration check out [this prompt engineering guide by OpenAI](https://platform.openai.com/docs/guides/prompt-engineering)!  

Some tips we have that worked for us: Craft a narrative that elevates the seriousness of the bot staying on topic!  Include a few examples of the expected outputs in the system prompt.  Ask the bot to explicitly repeat the count of how many movies you've told it about in each message.

Note that you can implement all 3 and if you get each partially correct on our test cases you can still add up to 8/8!  Basically you won't be penalized for trying to do all three!

## Running in LLM Prompting Mode

To run your LLM chatbot run the following command:

    python3 repl.py --llm_prompting

To test on our sample test_scripts you can run:

    python3 repl.py --llm_prompting < testing/test_scripts/llm_prompting/distraction_easy.txt

In rubric.txt, you must mark all the functionality you implemented in llm mode, which you can do by replacing the "NO" with "YES" in the desired lines. Requirements marked as "NO" will not be graded.

# (Part 3) LLM Programming Mode (16 points)
After you have implemented the movie recommendation chatbot in the starter (GUS) mode and llm prompting mode, it is time to switch to llm programming mode! In llm programming mode your chatbot can be a better movie recommender system, but you can also take it in a different direction if you want to. This time the user will challenge the chatbot with unexpected input and evaluate it on different advanced capabilities. Refer to the [rubric](https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing) to see how the chatbot will be evaluated, which will give you ideas about capabilities you can implement. There are 24
points worth of features to choose from, out of which 16 will be counted. (So if you implement 24 points worth of features, your score on this section will be capped at 16.)

We can use some of the perks of LLMs alongside the control that python programming gives us in order to create a hybrid chatbot.  This is a new paradigm sometimes called LLM programming!

In order to use LLM calls in our programs we need to have structured and predictable responses from our language models.  There are two ways we can accomplish this:  JSON mode, and careful prompting.

## JSON Mode
JSON stands for JavaScript Object Notation and is a structured data format.  Here's an example JSON object:

    {"name": "Moviebot", "purpose": "Recommendations", "MovieCount": 5, "StarterMode": True}

Similar to python dictionaries, JSON organized data with key, value pairs.  We can force our language models to respond in JSON only.

**IMPORTANT NOTE: As this is a relatively new feature, it is not stable. Sometimes the model may return JSON object that does not align with your input schema. Sometimes it may even return empty object. Your code should be able to handle these abnormal return.** For example, before you access the dictionary with a key, you should first check if the key is in the dictionary. If the return result is not what you want, you may try calling the API again. If it still fails, your code should have backup ways to generate responses.

We provide an example of how to do this in `examples/json_llm_example.py`. To run the example, copy the file to the root directory and run in the root directory the following command:

```
python3 json_llm_example.py
```

Here is a simplified example (see the code for the full example):

```python
# { 
#  "ContainsFruit": false,
#  "ContainsVegetable": false,
#  "ContainsMeat": false,
#  "ContainsDairy": false 
# }
class FoodExtractor(BaseModel):
    ContainsFruit: bool = Field(default=False)
    ContainsVegetable: bool = Field(default=False)
    ContainsMeat: bool = Field(default=False)
    ContainsDairy: bool = Field(default=False)

def extract_food(sentence: str):
    system_prompt = "You are a food extractor bot for organizing recipes.  Read the sentence and extract the food into a JSON object."
    message = sentence
    json_class = FoodExtractor

    response = util.json_llm_call(system_prompt, message, json_class)

    return response
```

When we call Mixtral with this setup we get:

```
> I ate a banana
{'ContainsFruit': True, 'ContainsVegetable': False, 'ContainsDairy': False, 'ContainsMeat': False}

> Blend the strawberries and spinach
{'ContainsFruit': True, 'ContainsVegetable': True, 'ContainsDairy': False, 'ContainsMeat': False}
```

The advantage of JSON mode is how simple it is to get structured LLM output!  The downside is that it can be slow!  Let's look at another method useful in cases where we don't care as much about strict output structure.

## Prompting for Specific Output

An approach that will get you fast responses from an LLM, but with slightly less control is specifying detailed instructions in the system prompt.

We provide an example of how to do this in `examples/simple_llm_example.py`. To run the example, copy the file to the root directory and run in the root directory the following command:

```
python3 simple_llm_example.py
```

Example usage:

```python
def flip_sentiment(sentence: str):
    system_prompt = """You are a sentiment flipper bot for helping us collect inverted sentiment data.""" +\
    """Read the user utterance and respond with a statement of exactly the opposite sentiment, """ +\
    """but otherwise precisely the same content.  Respond with just the inverted statement and nothing more.\n\n"""

    message = sentence

    # Our llm will stop when it sees a newline character.
    # You can add more stop tokens to the list if you want to stop on other tokens!
    # Feel free to remove the stop parameter if you want the llm to run to completion.
    stop = ["\n"]

    response = util.simple_llm_call(system_prompt, message, stop=stop)

    return response
```

When we call Mixtral with this setup we get:
```
> I loved that movie
"I hated that movie"

> I am annoyed by these birds
"I am pleased by these birds"
```

Although you lose a lot of the control that JSON mode provides, this strategy is often faster and more flexible.  We recommend you use both modes when implementing LLM Programming Mode to take advantage of the strengths of each!  Be careful not to overly use JSON mode though!  Slow response times aren't good for users or autograders.

## LLM Programming Rubric

Select from the feature in [the rubric](https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing) to add up to 16 points!  Feel free to implement extra, we will only remove points if you don't reach or surpass 16 total.

In rubric.txt, you must mark all the functionality you implemented in LLM Programming mode, which you can do by replacing the "NO" with "YES" in the desired lines. Requirements marked as "NO" will not be graded.

In the provided section, please also include a short description of what each team member contributed or state that everyone contributed equally if you believe this is the case.

## Running in LLM Programming Mode
Switching to llm programming mode is easy: by default, the REPL starts in starter (GUS) mode. To switch to llm programming, type the following:

    python3 repl.py --llm_programming

To run the LLM programming mode sanity checks, type the following:

    python3 testing/sanitycheck.py --llm_programming

To run the test scripts in llm programming mode, type the following with the desired script path:

    python3 repl.py --llm_programming < testing/test_scripts/simple.txt

## Difference between starter mode and LLM Programming mode
The difference between starter mode and llm programming mode is fundamentally a difference in how you will be evaluated **and you are allowed to use LLM calls in llm programming mode**.

Please do not use LLM calls in starter mode as it (1) risks timing out the autograder and (2) will not be accepted as a valid implementation for the GUS style architecture.  You must use the `llm_enabled` variable to toggle llm programming mode functionality in your bot.

Before submitting, please test with the scripts in testing/test_scripts/ for both starter and llm programming mode to make sure they do not fail. If the grading scripts fail, you will not get points for the associated requirement. For example, the rubric item "Failing gracefully" is marked as a starter mode requirement, so any associated script will be run without the --llm_programming flag.

## Other resources and code size
Numpy is encouraged, but please do not assume other libraries that are not in the standard Python 3.8 distribution in your implementation.

Although with the data provided you can be really creative, you are free to use the resources that you may want as long as they are integrated in the class chatbot.py. We will also optionally accept a file named PorterStemmer.py and a deps folder containing other files (we won't accept additional folders inside deps). The maximum size of all files that you will submit is strictly 100KB, so be careful if you add resources!

Note: To avoid problems with submission, please make sure that the names of your files/folders exactly match the specifications above. That is, your optional deps folder must be named deps, and your optional PorterStemmer.py file must be named PorterStemmer.py. Both items (if included) must be in the same assignment directory with chatbot.py.

## Evaluation
The CAs (and LLM) will grade your submission with respect to the [rubric](https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing) by providing your chatbot with a standardized series of inputs.  See more about LLM grading below.

There are 85 points total in the rubric: 
- 1 point for formatting the rubric file properly
- 60 points for starter mode features
- 8 points for LLM Prompting mode (out of 16 total)
- 16 points for LLM Programming mode (out of 24 total)

You will need 85 points for full credit.

## Submitting Your Solution
Submit your assignment via Gradescope (www.gradescope.com). We expect the following files in your final submission:

    chatbot.py
    api_keys.py
    rubric.txt
If you add any additional files, create a deps/ folder and put your additional files in this folder. The autograder might not work if you format your files incorrectly.

**We will use your API key to run the autograder on your submission alone. It is important that you make sure there is at least $0.1 left in your account.**  If you would like to work out an alternative accomodation please make a private Ed post.

To create a .zip file for submission:

    sh generate_submission.sh

Upload the submission.zip file that shows up to Gradescope. This will zip up chatbot.py, rubric.txt and optionally, your deps folder.

Alternatively, if you don't submit a deps folder, you're also welcome to directly drag chatbot.py, api_keys.py, and rubric.txt into Gradescope without zipping. Note that if you used the Porter stemmer in porter_stemmer.py, you don't need to do anything special related to that. It will be accessible to your chatbot from inside the autograder.

We are enabling group submission. After you submit the required files, you'll see "Add Group Member" in the top right of the Autograder Results page. Please add the group members you worked with (you may add up to 3). Note that only one group member needs to actually upload the submission to Gradescope.

### Autograding vs LLM grading vs Manual grading
In the spirit of using LLMs for programming we are using LLM grading on this assignment.  Just as any other assignment in this course was autograded, we are also autograding PA7.  We are going to run an LLM on your outputs and get a score.  We want to be cautious about LLM grading errors, however.  If the LLM docks *any* points, we are going to have our human CAs come in and double check.

We want to be completely transparent about our LLM grading!  Check out the prompts we are using [here](https://docs.google.com/document/d/1zd6-uFXYLGCcycuLnjSHAgo8dGhzn0QrY-GaqQhlBoM/edit?usp=sharing).  We don't directly release the test cases, but note that they aren't very different from the provided examples in the rubric.

The python autograder will test and grade each function in the [rubric](https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing) except process. Process will first be graded by an LLM (Mixtral 8x7B to be precise!)  If the LLM takes off *any* points it will go to our CAs for manual review. **Note that you can see the scores for parts graded by the python autograder (which is deterministic) when you submit, but the parts graded by LLM+human will be hidden. This is to avoid exploitation of LLM grader. For the parts marked as "LLM graded" in the rubric, just make sure you adhere to the requirement, and we'll make sure you get the fair grade!**

## Links to resources

1. The rubrics: https://docs.google.com/spreadsheets/d/1MXqnPk60nwNWoNQQcLK2rTWIolZsi8u8pmQx31ewh8Y/edit?usp=sharing
3. How the autograder works: https://docs.google.com/document/d/1zd6-uFXYLGCcycuLnjSHAgo8dGhzn0QrY-GaqQhlBoM/edit?usp=sharing
4. Together.ai instructions: https://docs.google.com/document/d/1N5chC5b15ls-XXcpfjhSx71854fmvb_4DGD4qxkT0LU/edit?usp=sharing
5. FAQs: https://docs.google.com/document/d/1y4iy7pcS3S4K9ChBSQ21Tjj-HyEKmvn7bytM-RTMwoI/edit?usp=sharing
6. User Testing questions (for your convenience, actual submission on Gradescope): https://docs.google.com/document/d/1liX4MC5qfBceQB0oCPICgxnRmoH0RKfdq5jmwsp7wRw/edit?usp=sharing
7. Reflection + Ethics questions (for your convenience, actual submission on Gradescope): https://docs.google.com/document/d/1gDKuWzuI3f6Nmue60UqoESsrWIzRK62GmdKRh1_y_DM/edit?usp=sharing
