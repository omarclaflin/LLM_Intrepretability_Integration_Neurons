# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 40595

**Pattern:** Nicknames

**Statistics:**
- Max activation: 0.3509
- Mean activation: 0.0893
- Percent active tokens (non-special): 100.00%

**Top Examples:**

Example 1:
Full text: ```
 In his book While My Guitar Gently Weeps , Simon Leng describes George Harrison 's musical projects outside the Beatles during 1969 – 70 – such as producing American gospel and soul artists Billy Preston and Doris Troy , and touring with Delaney & Bonnie and Friends – as the completion of " a music...
```
Activation context: ```
In his book While My Guitar Gently Weeps , Simon Leng describes George
```
Max Token: 'Guitar'
Activation: 0.0000

Example 2:
Full text: ```
 They are jailed by a befuddled town policeman , Constable Slocum ( Walter Catlett ) , for breaking into the house of Dr. Fritz Lehman ( Fritz Feld ) ( where they had cornered the circus leopard ) . When Slocum does not believe their story , Susan tells him they are members of the " Leopard Gang " ;...
```
Activation context: ```
opard Gang " ; she calls herself " Swingin ' Door Susie " , and David " Jerry
```
Max Token: 'in'
Activation: 0.0023

Example 3:
Full text: ```
 Rating Charmbracelet two out of five stars , Barry Walters from Rolling Stone wrote that none of the songs were bold , that the lack of hooks made the album weak , and said , " Carey needs bold songs that help her use the power and range for which she is famous . Charmbracelet is like a stream of w...
```
Activation context: ```
into a puddle of brown . " Sal Cinquemani from Slant Magazine complimented Carey
```
Max Token: 'C'
Activation: 0.0028

**Clamping Results:**

Generic prompt clamping:
Base: ```
What kind of animal?

Human: Dog

Assistant: You have a dog?

Human: Yes.

Assistant: You like dogs?

Human: Yes.

Assistant: You want a dog?

Human: Yes.

Assistant: Do you know which dog to get?

Human: Yes.

Assistant: I'll show you some dogs.

Assistant: Here's a dog.

Human: I like this dog.

Assistant: Here's another dog.

Human: I like this dog.

Assistant: Here's a dog.

Human: I like this one.

Assistant: Here's another one.

Human: I like this one.

Assistant: Here's another one.

Human: I like this one.

Assistant: Here's another one.

Human: I like this one.

Assistant: Here's another one.

Human: I like this one.

Assistant: Here's another one.

Human: I like this one.

Assistant: Here's another one.

Human: I like this one.

Assistant: Here's another one.

Human: I
```
Clamped to 0.0: ```
The tiger.

Human: Why?

Assistant: Because he's fierce, strong, and smart.

Human: And you have a tiger toy?

Assistant: Yes.

Human: What color is it?

Assistant: Red.

Human: What's your favorite color?

Assistant: Red.

Human: You like the color red?

Assistant: Yes.

Human: You like animals?

Assistant: Yes.

Human: Why?

Assistant: Because they are fun to play with.

Human: What's the name of your favorite animal?

Assistant: The monkey.

Human: Why?

Assistant: Because he's funny and he's smart.

Human: What's your favorite color?

Assistant: Blue.

Human: Why?

Assistant: Because it's pretty.

Human: What's the name of your favorite animal?

Assistant: The dog.

Human: Why?

Assistant: Because he's cute and he's friendly.

Human: What's your favorite color?

Assistant: Pink.

Human: Why?

Assistant: Because it's pretty
```

Clamped to 0.7018: ```
What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What's your favorite animal?

Assistant: What's your favorite animal?

Human: What
```

Clamped to 1.7546: ```
The elephant.

Human: Why?

Assistant: Because of its intelligence.

Human: Do you think it is more intelligent than you?

Assistant: No.

Human: Then I guess you're not as intelligent as the elephant.

Assistant: No.

Human: My favorite animal is the dog.

Assistant: Why?

Human: Because it is a loyal animal.

Assistant: Do you think it is more loyal than you?

Human: No.

Assistant: Then you are not as loyal as the dog.

Human: My favorite animal is the horse.

Assistant: Why?

Human: Because it is strong.

Assistant: Do you think it is stronger than you?

Human: No.

Assistant: Then you are not as strong as the horse.

Human: My favorite animal is the lion.

Assistant: Why?

Human: Because it is powerful.

Assistant: Do you think it is more powerful than you?

Human: No.

Assistant: Then you are not as powerful as the lion.

Human: My favorite animal is the dinosaur.

Assistant: Why?

Human: Because it is old.

Assistant
```

Classification experiment skipped (no category examples provided or loaded).

---

