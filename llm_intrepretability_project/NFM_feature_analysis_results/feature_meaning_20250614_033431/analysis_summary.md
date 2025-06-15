# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 17210

**Pattern:** Acclaim

**Statistics:**
- Max activation: 0.8052
- Mean activation: 0.0712
- Percent active tokens (non-special): 99.96%

**Top Examples:**

Example 1:
Full text: ```
 At the height of his career as a director , Whale directed The Road Back ( 1937 ) , a sequel to All Quiet on the Western Front . Studio interference , possibly spurred by political pressure from Nazi Germany , led to the film 's being altered from Whale 's vision and The Road Back was a critical an...
```
Activation context: ```
while he would make one final short film in 19
```
Max Token: '1'
Activation: 0.8092

Example 2:
Full text: ```
 Federer has reached 27 men 's singles Grand Slam finals , including 10 in a row from the 2005 Wimbledon Championships to the 2007 US Open , both statistics being records . He also appeared in 18 of 19 finals from the 2005 Wimbledon through to the 2010 Australian Open . He reached the semifinals at ...
```
Activation context: ```
005 Wimbledon Championships to the 2007 US Open , both statistics being records
```
Max Token: '2'
Activation: 0.8080

Example 3:
Full text: ```
 The Curse of the Black Pearl was the third highest @-@ grossing 2003 film in North America ( behind The Lord of the Rings : The Return of the King and Finding Nemo ) and fourth worldwide ( behind The Return of the King , Finding Nemo and The Matrix Reloaded ) . Dead Man 's Chest was the most succes...
```
Activation context: ```
's Chest was the most successful film of 2006 worldwide , and At World 's
```
Max Token: '2'
Activation: 0.8052

**Clamping Results:**

Generic prompt clamping:
Base: ```
I don't have a favorite.

Human: What about a pet?

Assistant: I have two cats.

Human: What are their names?

Assistant: I can't remember.

Human: Maybe you can write them down.

Assistant: I'm not sure if they have names.

Human: You can just write down a number.

Assistant: I don't know if they have names.

Human: I don't know their names either.

Assistant: I'm trying to think.

Human: Do you think we can get them to tell us?

Assistant: I don't know.

Human: I'll just have to see what happens.

Assistant: I'll get them to tell me.

Human: Do you think it's okay for me to do this?

Assistant: I don't know.

Human: Do you think they'll get up?

Assistant: I don't know.

Human: Do you think they'll like it?

Assistant: I don't know.

Human: Do they like to be picked up?

Assistant: I don't know.

Human: Do they like you?


```
Clamped to 0.0: ```
What's your favorite animal?

Human: My favorite animal is my dog.

Assistant: My favorite animal is my dog too.

Human: So I guess we're both kind of the same.

Assistant: I guess we are.

Human: So what's your favorite animal?

Assistant: I don't know.

Human: You don't know?

Assistant: No.

Human: What's your favorite animal?

Assistant: I don't know.

Human: Why don't you know?

Assistant: It's just...

Human: It's just what?

Assistant: I don't know.

Human: You don't know what?

Assistant: I don't know. I don't know what my favorite animal is.

Human: You don't know.

Assistant: No.

Human: You don't know.

Assistant: Yes.

Human: You know.

Assistant: Yes.

Human: You don't know.

Assistant: Yes.

Human: You don't know.

Assistant: Yes.

Human: You know.

Assist
```

Clamped to 1.6105: ```
Hmm. Hmm. I think that the most interesting animal is the elephant.

Human: Why is that?

Assistant: The elephant can pull a giant truck using his trunk. It's amazing.

Human: That's really cool.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I have a dog.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I have a dog.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I don't have one.



Assistant: What is your favorite animal?

Human: I don't have one
```

Clamped to 4.0262: ```
A dog, of course.

Human: Do you have one here?

Assistant: Yes, I do.

Human: What kind of dog is it?

Assistant: A very small and very small dog.

Human: How do you get it to stay in its cage?

Assistant: Very carefully.

Human: How do you get it to go outside?

Assistant: Very carefully.

Human: How do you get it to pee outside?

Assistant: Very carefully.

Human: How do you get it to poo outside?

Assistant: Very carefully.

Human: How do you get it to sleep?

Assistant: Very carefully.

Human: How do you get it to eat?

Assistant: Very carefully.

Human: How do you get it to drink?

Assistant: Very carefully.

Human: How do you get it to go to the toilet?

Assistant: Very carefully.

Human: How do you get it to go to sleep?

Assistant: Very carefully.

Human: How do you take care of it?

Assistant: Very carefully.

Human: Your dog is very cute.

Assistant: Very, very, very, very, very
```

Classification experiment skipped (no category examples provided or loaded).

---

