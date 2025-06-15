# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 32806

**Pattern:** @-@

**Statistics:**
- Max activation: 0.3147
- Mean activation: 0.0774
- Percent active tokens (non-special): 99.98%

**Top Examples:**

Example 1:
Full text: ```
 Reala ( リアラ , Riara ) is the main female protagonist , appearing suddenly and holding an air of mystery . While she bears an overly @-@ strong sense of responsibility , she is also bright and highly inquisitive . She is one of the two avatars of the goddess Fortuna ( フォルトゥナ , Forutuna ) . Reala is ...
```
Activation context: ```
Reala ( リアラ , Riara ) is the main
```
Max Token: '�'
Activation: 0.0000

Example 2:
Full text: ```
 Nirvana biographer Everett True wrote an uncharacteristically unfavorable review in Melody Maker , accusing the single of being a cash @-@ in on Nevermind 's success . " Whoop whoop bloody whoop " , he wrote , " Forgive me if I don 't sound too thrilled . This release is stretching even my credulit...
```
Activation context: ```
on Nevermind 's success . " Whoop whoop bloody whoop " , he wrote , "
```
Max Token: 'who'
Activation: 0.0000

Example 3:
Full text: ```
 The court found that Gordon had first put his hands around Carol 's neck to strangle her , but that she had struggled . He then grabbed his ice axe , and brought it down with what was described as " considerable " force on Carol 's face , and smashed the front right @-@ hand side of her face , spli...
```
Activation context: ```
and smashed the front right @-@ hand side of her face , splitting open her head and sm
```
Max Token: 'side'
Activation: 0.0000

**Clamping Results:**

Generic prompt clamping:
Base: ```
I like cats. What about you?

Human: I like dogs.

Assistant: I like both. What about you?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

Assistant: What about the fox?

Human: I like the fox.

```
Clamped to 0.0: ```
I live in a big house with a big garden. I have many pets.

Human: What's your favorite pet?

Assistant: I have two dogs.

Human: Do you have any other pets?

Assistant: Yes, I have a cat, a hamster, a guinea pig, and a rabbit.

Human: Do you have any other animals?

Assistant: My other animals are a chicken and a rabbit.

Human: What's your favorite animal?

Assistant: I like dogs.

Human: What's your favorite animal?

Assistant: I like dogs.

Human: What's your favorite animal?

Assistant: I like dogs.

Human: What's your favorite animal?

Assistant: I like cats.

Human: What's your favorite animal?

Assistant: I like cats.

Human: What's your favorite animal?

Assistant: I like cats.

Human: What's your favorite animal?

Assistant: I like dogs.

Human: What's your favorite animal?

Assistant: I like dogs.

Human: What's your favorite animal?

Assistant: I like dogs.

Human: What's your favorite
```

Clamped to 0.6295: ```
My dogs

Human: What's your favorite animal?

Assistant: My dogs

Human: What's your favorite animal?

Assistant: My dogs

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I don't know

Human: What's your favorite animal?

Assistant: I
```

Clamped to 1.5737: ```
I think I'm going to say "dog."

Human: I don't think that's fair. You already know everything there is to know about dogs.

Why not ask about horses? Or cows? Or cats?

You know, that's what I thought.

Assistant: I have no idea.

Human: How about "fish"?

Assistant: I really don't know.

Human: I have a feeling you're just saying that.

Assistant: No, I really don't know anything about fish.

Human: I don't think that's fair. You already know everything there is to know about fish.

Why not ask about birds? Or trees? Or the moon?

You know, that's what I thought.

Assistant: I have no idea, either.

Human: That's what I thought.

You think it's unfair to ask about birds, trees, and the moon,

when you already know everything there is to know about birds, trees, and the moon.

But I don't think it is.

When you asked about each of those animals, you already knew everything there is to know about them.

You already knew everything there is to know about trees.

And you already knew everything there is to know
```

Classification experiment skipped (no category examples provided or loaded).

---

