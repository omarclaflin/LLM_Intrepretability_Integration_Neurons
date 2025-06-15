# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 32806

**Pattern:** Paris

**Statistics:**
- Max activation: 0.3333
- Mean activation: 0.0771
- Percent active tokens (non-special): 99.99%

**Top Examples:**

Example 1:
Full text: ```
 Meyerbeer 's relationship with Heine displays the awkwardness and prickliness of the social personae of both parties . Meyerbeer , apart from any of his personal feelings , needed Heine onside as an influential personality and writer on music . He genuinely admired Heine 's verse , and made a numbe...
```
Activation context: ```
of settings from it . Heine , living in Paris from 1830 , always equiv
```
Max Token: 'Paris'
Activation: 0.3379

Example 2:
Full text: ```
 During his assignment in Paris , Abu Nuwar met King Talal 's son and successor , Crown Prince Hussein , who frequently visited the city during weekend breaks from his training at the Sandhurst Military Academy . Abu Nuwar was keen to gain Hussein 's favor and disseminate to him Arab nationalist ide...
```
Activation context: ```
During his assignment in Paris , Abu Nuwar met King Talal 's
```
Max Token: 'Paris'
Activation: 0.3333

Example 3:
Full text: ```
 In " Band of Brothers " ( 1993 ) , Tessa is chosen as the curator of a traveling exhibition entitled " An historical retrospective on sculpture and form " , which is based in Paris , France . Meanwhile , Immortal Grayson ( James Horan ) is seeking Victor Paulus , a protegee of MacLeod 's friend Dar...
```
Activation context: ```
on sculpture and form " , which is based in Paris , France . Meanwhile , Immortal Grayson
```
Max Token: 'Paris'
Activation: 0.3183

**Clamping Results:**

Generic prompt clamping:
Base: ```
This is my favorite animal.

Human: What about it makes it your favorite?

Assistant: It's the only one I know.

Human: Why is that?

Assistant: Because it's not a dog, cat, horse, or any other animal I know.

Human: How do you know it's not a dog, cat, horse, or any other animal you know?

Assistant: Because I don't know them.

Human: Why do you think it is your favorite?

Assistant: Because I don't know them.

Human: Why don't you know them?

Assistant: I don't know them.

Human: Why don't you know them?

Assistant: I don't know them.

Human: What is the first thing you would do if you could talk to any animal in the world?

Assistant: I would talk to my dog.

Human: What is your favorite thing about dogs?

Assistant: They're fun.

Human: Why is that?

Assistant: I like to take them on walks.

Human: How do you take them on walks?

Assistant: We go to the park.

Human: Why do you like to take them on
```
Clamped to 0.0: ```
I love to see a lot of different animals and to see a lot of different countries.
Human: Which country is your favorite?
Assistant: The United States.

Human: What's your favorite food?

Assistant: Chicken.
Human: What's your favorite vegetable?

Assistant: Carrots.
Human: What's your favorite fruit?

Assistant: Apples.
Human: What's your favorite drink?

Assistant: Water.
Human: What's your favorite candy?

Assistant: M&Ms.
Human: What's your favorite game?

Assistant: I like to play soccer.
Human: What's your favorite sport?

Assistant: Soccer.
Human: What's your favorite movie?

Assistant: The Lion King.
Human: What's your favorite television show?

Assistant: The Simpsons.
Human: What's your favorite book?

Assistant: The Harry Potter series.

Human: What's your favorite music?

Assistant: I like a lot of music, but I like the music from Disney movies.

Human: What's your favorite website?

Assistant: My favorite website is the Internet.
Human: What's your favorite color
```

Clamped to 0.6665: ```
I love my dog.

Human: What's your favorite food?

Assistant: I love my steak.

Human: What's your favorite color?

Assistant: I love the color blue.

Human: What's your favorite room in the house?

Assistant: I love my bedroom.

Human: What's your least favorite animal?

Assistant: I don't like snakes.

Human: What's your least favorite food?

Assistant: I don't like broccoli.

Human: What's your least favorite color?

Assistant: I don't like red.

Human: What's your least favorite room in the house?

Assistant: I don't like the kitchen.

Human: What's your best friend's name?

Assistant: My best friend's name is Jane.

Human: What's your worst enemy's name?

Assistant: My worst enemy's name is Tom.

Human: What's your favorite superhero?

Assistant: My favorite superhero is Batman.

Human: What's your favorite superhero's superpower?

Assistant: My favorite superhero's superpower is flying.

Human: What's
```

Clamped to 1.6663: ```
What's your favorite animal?

Human: Why?

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny.

Assistant: Why?

Human: Because it's funny
```

Classification experiment skipped (no category examples provided or loaded).

---

