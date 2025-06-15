# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 23360

**Pattern:** released

**Statistics:**
- Max activation: 0.2282
- Mean activation: 0.0626
- Percent active tokens (non-special): 99.96%

**Top Examples:**

Example 1:
Full text: ```
 " In Bloom " was released as the fourth single from Nevermind on November 30 , 1992 . The single was only released commercially in the United Kingdom ; promotional copies were released in the United States . The 7 @-@ inch vinyl and cassette editions of the single contained a live version of " Poll...
```
Activation context: ```
the 12 @-@ inch vinyl and CD versions featured a performance of " Sliver "
```
Max Token: 'CD'
Activation: 0.0000

Example 2:
Full text: ```
 Kedok Ketawa was released in Batavia in July 1940 , with a press screening on 20 July . By September it was being shown in Surabaya . In some newspaper advertisements , such as in Pemandangan , it was referred to as Pendekar dari Preanger ( Warrior from Preanger ) , while in others it was advertise...
```
Activation context: ```
andangan , it was referred to as Pendekar dari Preanger ( Warrior from Pre
```
Max Token: 'end'
Activation: 0.0000

Example 3:
Full text: ```
 Giacomo Meyerbeer ( born Jacob Liebmann Beer ; 5 September 1791 – 2 May 1864 ) was a German opera composer of Jewish birth who has been described as perhaps the most successful stage composer of the nineteenth century . With his 1831 opera Robert le diable and its successors , he gave the genre of ...
```
Activation context: ```
With his 1831 opera Robert le diable and its successors , he gave the genre
```
Max Token: 'di'
Activation: 0.0000

**Clamping Results:**

Generic prompt clamping:
Base: ```
What's that?

Human: The giraffe.

Assistant: What's it like?

Human: It's got a cute little face with a long neck.

Assistant: What's it do?

Human: It eats grass.

Assistant: What's it look like?

Human: It looks like a small horse.

Assistant: What's it's name?

Human: Giraffe.

Assistant: What's it's color?

Human: It's got a purple coat.

Assistant: What's that?

Human: It's got a stripe-y face.

Assistant: What's it's belly?

Human: It's got a big purple belly.

Assistant: What's it's eyes?

Human: It's got a big round face.

Assistant: What's it's mouth?

Human: It's got a black mouth.

Assistant: What's it's nose?

Human: It's got a long pink nose.

Assistant: What's it's feet?

Human: It's got a cute little pink toes.

Assist
```
Clamped to 0.0: ```
Cats

Human: Cats are great, but I have a special place in my heart for dogs.

Assistant: Dogs are great, but I have a special place in my heart for cats.

Human: Cats and dogs are great. But I have a special place in my heart for reptiles.

Assistant: Reptiles are great, but I have a special place in my heart for birds.

Human: Birds are great, but I have a special place in my heart for dolphins.

Assistant: Dolphins are great, but I have a special place in my heart for whales.

Human: Whales are great, but I have a special place in my heart for humans.

```

Clamped to 0.4564: ```
Oh, I love to watch birds fly.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird's a bird?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Assistant: I do.

Human: Do you know what a bird is?

Ass
```

Clamped to 1.1410: ```
A cow.

Human: (mournfully) Good. Now, what's your favorite color?

Assistant: Green.

Human: (mournfully) Good. Now, what's your favorite food?

Assistant: (glaring) Bugs.

Human: (mournfully) Good. Now, what's your favorite color?

Assistant: (glaring) Blue.

Human: (mournfully) Good. Now, what's your favorite food?

Assistant: (glaring) Bugs.

Human: (mournfully) Good. Now, what's your favorite color?

Assistant: (glaring) Red.

Human: (mournfully) Good. Now, what's your favorite food?

Assistant: (glaring) Bugs.

Human: (mournfully) Good. Now, what's your favorite color?

Assistant: (glaring) Red.

Human: (mournfully) Good. Now, what's your favorite food?

Assistant: (glaring) Bugs.

Human: (mournfully) Good. Now, what's your favorite color?

Assistant: (glaring) Red.

Human: (mournfully)
```

Classification experiment skipped (no category examples provided or loaded).

---

