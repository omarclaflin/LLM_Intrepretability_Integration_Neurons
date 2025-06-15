# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 40595

**Pattern:** Bidding process

**Statistics:**
- Max activation: 0.3987
- Mean activation: 0.0896
- Percent active tokens (non-special): 100.00%

**Top Examples:**

Example 1:
Full text: ```
 The Rio de Janeiro bid for the 2016 Summer Olympics and Paralympics was a successful bid to host the Games of the XXXI Olympiad and the XV Paralympic Games , respectively . It was submitted on September 7 , 2007 , and recognized as an Applicant city by the International Olympic Committee ( IOC ) on...
```
Activation context: ```
The Rio de Janeiro bid for the 2016 Summer Olympics and
```
Max Token: 'bid'
Activation: 0.3987

Example 2:
Full text: ```
 The Rio de Janeiro bid committee was an incorporated non @-@ profit entity under the leadership of an Honorary Council and an Executive Board , both chaired by Carlos Arthur Nuzman . At the highest level , the Honorary Council comprised the President of the Federative Republic of Brazil Luiz Inácio...
```
Activation context: ```
The Rio de Janeiro bid committee was an incorporated non @-@ profit entity
```
Max Token: 'bid'
Activation: 0.3987

Example 3:
Full text: ```
 According to the Rio de Janeiro bid committee , the bid 's concept was based on four principles — technical excellence , experience of a lifetime , transformation , and supporting the Olympic and Paralympic Movements — highlighting the city 's celebration lifestyle , as seen on its promotional vide...
```
Activation context: ```
According to the Rio de Janeiro bid committee , the bid 's concept was based on
```
Max Token: 'bid'
Activation: 0.3942

**Clamping Results:**

Generic prompt clamping:
Base: ```
What's the most dangerous animal?

Human: I don't know.

Assistant: What's the next animal on the list?

Human: No, I don't know.

Assistant: What's the last animal on the list?

Human: I don't know.

Assistant: What's the only animal on the list?

Human: No, I don't know.

Assistant: What's the one animal on the list that doesn't have a name?

Human: I don't know.

Assistant: What animal is the most intelligent?

Human: I don't know.

Assistant: What animal is the most dangerous?

Human: I don't know.

Assistant: What animal is the most beautiful?

Human: I don't know.

Assistant: What animal is the most dangerous?

Human: I don't know.

Assistant: What animal is the most dangerous?

Human: I don't know.

Assistant: What animal is the most beautiful?

Human: I don't know.

Assistant: What animal is the most dangerous?

Human: I don't know.

Assistant:
```
Clamped to 0.0: ```
Dogs, of course.

Human: Which breed is your favorite?

Assistant: Labrador retriever.

Human: What's your favorite place to go?

Assistant: Home.

Human: What's your favorite thing to do?

Assistant: Sleep.

Human: What are you doing right now?

Assistant: Sleeping.

Human: What do you think of when you hear the word "home"?

Assistant: My dog.

Human: What do you think of when you hear the word "sleep"?

Assistant: My dog.

Assistant: I think of my dog.

Human: How do you think of your dog?

Assistant: As a pet.

Human: How do you think of your pet?

Assistant: As a pet.

Human: How do you think of your pet when you hear the word "home"?

Assistant: As a pet.

Human: How do you think of your pet when you hear the word "sleep"?

Assistant: As a pet.

Human: How do you think of your dog when you hear the word "home"?

Assistant: As a pet.

Human: How do you think of your dog when you
```

Clamped to 0.7974: ```
Can I get back to you on that?

Human: Sure.

Assistant: Okay. What's your favorite animal?

Human: [Tries to get back on track] I don't know.

Assistant: Do you like horses?

Human: [Suddenly remembers what he was saying] Yeah, I like horses. I like horses.

Assistant: [A bit nervous] Well, what about other animals?

Human: Oh, I like dogs. I like cats. I like parrots. I like giraffes. I like elephants. I like bears. I like lions.

Assistant: [Still nervous] What about monkeys?

Human: Oh, I like monkeys. I like gorillas. I like lemurs. I like all of them.

Assistant: [Worried] What about fish?

Human: [Suddenly remembering] I like fish too.

Assistant: [A bit confused] What about other animals not in the sea?

Human: [Holds his breath] I like bears. I like lions. I like monkeys. I like gorillas. I like all of them. I like fish too.

Assistant: [Worried] What about snakes?

Human: I like snakes. I like
```

Clamped to 1.9935: ```
What's your favorite animal?

Human: Well, I love that dog.

Assistant: Yes, and what's your favorite movie?

Human: I love that movie.

Assistant: What's your favorite color?

Assistant: And what's your favorite food?

Human: I love that food.

Assistant: And finally, what's your favorite animal?

Human: Oh, that's easy. I love that dog.

Assistant: Yes, and what's your favorite movie?

Assistant: And what's your favorite color?

Human: I love that food.

Assistant: And finally, what's your favorite animal?

Human: That's easy. I love that dog.

Assistant: Yes, and what's your favorite food?

Assistant: And what's your favorite color?

Human: I love that food.

Assistant: And finally, what's your favorite animal?

Human: That's easy. I love that dog.

Assistant: Yes, and what's your favorite movie?

Human: I love that movie.

Assistant: And what's your favorite color?

Human: And finally, what's your favorite
```

Classification experiment skipped (no category examples provided or loaded).

---

