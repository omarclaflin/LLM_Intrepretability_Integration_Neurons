# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 17210

**Pattern:** Numbers

**Statistics:**
- Max activation: 0.7964
- Mean activation: 0.0713
- Percent active tokens (non-special): 99.96%

**Top Examples:**

Example 1:
Full text: ```
 Namco had high expectations for Destiny 2 , anticipating sales of 600 @,@ 000 units . During its debut week , Destiny 2 reached the top of Japanese gaming sales charts : according to different sources , it sold between just over 498 @,@ 000 ( Famitsu ) and under 558 @,@ 000 ( Dengeki ) units . For ...
```
Activation context: ```
over 498 @,@ 000 ( Famitsu ) and under 5
```
Max Token: '0'
Activation: 0.0000

Example 2:
Full text: ```
 The details of the murder are sketchy . Carol was killed due to blunt trauma to her face by means of some instrument , alleged in court to have been an ice axe . She was then bound with rope , using complex knots , weighed down with rocks and lead pipes and thrown overboard from a boat on Coniston ...
```
Activation context: ```
amateur divers . Had it been dropped a few metres further from the shore , it would have
```
Max Token: 'further'
Activation: 0.0000

Example 3:
Full text: ```
 The song received mixed reviews from critics . Azeem Ahmad of OMH Media proposed to ban " My Boo " because he felt that the collaboration was a competition among Usher and Alicia Keys " over the hip @-@ pop like beat " . Jon Caramanica of Rolling Stone found the duet sentimental . At the 2005 Gramm...
```
Activation context: ```
R & B Performance by a Duo or Group with Voc
```
Max Token: 'with'
Activation: 0.0000

**Clamping Results:**

Generic prompt clamping:
Base: ```
My favorite animal is the dog.

Human: What's your favorite dog?

Assistant: The golden retriever.

Human: What's your favorite golden retriever?

Assistant: This is a golden retriever.

Human: What's this dog's name?

Assistant: This is a golden retriever.

Human: What's this dog's name?

Assistant: This is a gold-

Human: Oh, I get it.

10

Human: What's your favorite book?

Assistant: My favorite book is the Iliad.

Human: What's your favorite book in the Iliad?

Assistant: The Iliad.

Human: What's the Iliad?

Assistant: It's the story of the Trojan war.

Human: What's the Trojan war?

Assistant: It was fought 3000 years ago.

Human: What's the Trojan war?

Assistant: It was fought 3000 years ago.

Human: So it's a story about the Trojan war?

Assistant: Yes.

Human: So it's a story about a war
```
Clamped to 0.0: ```
What's my favorite animal?

Human: [whispers] Sheep.

The Assistant: [whispering] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [whispers] Sheep?

Human: [whispers] Sheep.

The Assistant: [
```

Clamped to 1.5928: ```
The giraffe.

Human: Why?

Assistant: Because it has a long neck.

Human: What's your favorite color?

Assistant: The blue.

Human: Why?

Assistant: Because it's the king of the animals and it doesn't look like any other animal.

Human: What type of bird is that?

Assistant: It's a sparrow.

Human: Why?

Assistant: Because it's a small bird that has a long beak.

Human: What type of plant is that?

Assistant: It's an ivy.

Human: Why?

Assistant: Because it's a small plant that has a long root.

Human: What type of animal is that?

Assistant: It's a dog.

Human: Why?

Assistant: Because it's the king of the animals and it doesn't look like any other animal.

Human: What type of fish is that?

Assistant: It's a salmon.

Human: Why?

Assistant: Because it's a small fish that has a long tail.

Human: What type of tree is that?

Assistant: It's an oak
```

Clamped to 3.9819: ```
What's the name of the animal?

Human: The monkey.

Assistant: Monkey?

Human: The monkey.

Assistant: Monkey?

Human: The monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.

Assistant: Monkey?

Human: Monkey.


```

Classification experiment skipped (no category examples provided or loaded).

---

