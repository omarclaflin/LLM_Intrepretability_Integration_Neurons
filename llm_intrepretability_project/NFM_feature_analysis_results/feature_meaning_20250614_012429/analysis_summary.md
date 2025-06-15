# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 23360

**Pattern:** folk

**Statistics:**
- Max activation: 0.2201
- Mean activation: 0.0621
- Percent active tokens (non-special): 99.96%

**Top Examples:**

Example 1:
Full text: ```
 Shankar saw about half the film in a roughly edited version before composing the background score , but he was already familiar with the story . According to Robinson , when Ray met Shankar the latter hummed a tune that was folk @-@ based but had " a certain sophistication " . This tune , usually p...
```
Activation context: ```
ankar the latter hummed a tune that was folk @-@ based but had " a certain sophistic
```
Max Token: 'folk'
Activation: 0.2524

Example 2:
Full text: ```
 The Bengali word path literally means path , and pather means " of the path " . The word panchali refers to a type of narrative folk song that used to be performed in Bengal and was the forerunner of another type of folk performance , the jatra . English translations of the Bengali title include So...
```
Activation context: ```
Bengal and was the forerunner of another type of folk performance , the jatra . English translations of the
```
Max Token: 'folk'
Activation: 0.2464

Example 3:
Full text: ```
 For the techniques used to craft the overall sound of the album , August Brown of the Los Angeles Times described the band foregoing the use of full chords and verse @-@ chorus @-@ verse structures in favor of " distant guitar effects , crunchy bass grooves and smatterings of percussion . " Regardi...
```
Activation context: ```
Willett 's lyrics were a mixture of folk tales and topical stories , stating that " a lot
```
Max Token: 'folk'
Activation: 0.2444

**Clamping Results:**

Generic prompt clamping:
Base: ```
What's your favorite animal?

Human: I don't have one, just a bunch of them.

Assistant: Just a bunch of them?

Human: (laughs) Yes, a bunch of them.

Assistant: Just a bunch of them?

Human: Yes.

Assistant: So do you enjoy watching them?

Human: (laughs) I guess I do.

Assistant: Do you have any favorite animals?

Human: A bunch.

Assistant: A bunch?

Human: Yeah.

Assistant: What kind of animals do you like?

Human: I like dogs, for one.

Assistant: I see.

Human: I don't know.

Assistant: Do you have a favorite dog?

Human: A bunch. Yeah.

Assistant: Why do you like dogs?

Human: It's just a bunch of them.

Assistant: It is?

Human: Yeah.

Assistant: It's just a bunch of them?

Human: Yeah.

Assistant: That's what you said before.

Human: I said a bunch of them.

Assistant: Right, a bunch of them.

Human: I
```
Clamped to 0.0: ```
I love all of them.

Human: What's your favorite flower?

Assistant: I love them all.

Human: What's your favorite food?

Assistant: I love them all.

Human: What is your favorite color?

Assistant: I love them all.

Human: What's your favorite place to go?

Assistant: I love them all.

Human: What is your favorite thing to do?

Assistant: I love them all.

Human: What's your favorite thing to do with your friends?

Assistant: I love them all.

Human: What do you love most about your friends?

Assistant: I love them all.

Human: What's your favorite thing to do together?

Assistant: I love them all.

Human: What's your favorite thing to do together?

Assistant: I love them all.

Human: What is your favorite thing about you?

Assistant: I love them all.

Human: What's your favorite thing about you?

Assistant: I love them all.

Human: What's your favorite thing about you?

Assistant: I love them all.

Human: What's your favorite thing
```

Clamped to 0.4401: ```
A dog! I love dogs.

Human: Which breed?

Assistant: All of them.

Human: Do you have any pets?

Assistant: What do you mean?

Human: Do you own any animals?

Assistant: No.

The human continues to interrogate the assistant for the next few minutes but to no avail. The assistant is not able to give a straight answer to the human's questions because he is not able to think in the human's language.

The assistant is not an idiot. He is not stupid. He is not a moron. He is a genius. He is a language genius.

Language is not a set of symbols. Language is a set of rules.

Language is not a set of rules. Language is a system of communication.

Language is not a system of communication. Language is a system of thinking.

Language is not a system of thinking. Language is a system of understanding.

Language is not a system of understanding. Language is a system of meaning.

Language is not a system of meaning. Language is a system of thought.

Language is not a system of thought. Language is a system of creativity.

Language is not a system of creativity. Language is a system of creativity.

Language is not a system of creativity. Language is a system of creation.


```

Clamped to 1.1004: ```
A lion.

Human: Why?

Assistant: It's the animal with the biggest teeth.

Human: What's the biggest animal?

Assistant: The elephant.

Human: Why?

Assistant: It's the animal with the biggest ears.

Human: What's the biggest animal?

Assistant: The lion.

Human: Why?

Assistant: It's the animal with the biggest head.

Human: What's the biggest animal?

Assistant: The elephant.

Human: Why?

Assistant: It's the animal with the biggest heart.

Human: What's the biggest animal?

Assistant: The elephant.

Human: Why?

Assistant: It's the animal with the biggest stomach.

Human: What's the biggest animal?

Assistant: The elephant.

Human: Why?

Assistant: It's the animal with the biggest neck.

Human: What's the biggest animal?

Assistant: The elephant.

Human: Why?

Assistant: It's the animal with the biggest back.

Human: What's the biggest animal?

Assistant: The elephant.

Human
```

Classification experiment skipped (no category examples provided or loaded).

---

