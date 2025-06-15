# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 32806

**Pattern:** Numeric data

**Statistics:**
- Max activation: 0.3113
- Mean activation: 0.0767
- Percent active tokens (non-special): 99.97%

**Top Examples:**

Example 1:
Full text: ```
 = = = = Qatar Classic : 5 finals ( 5 titles , 0 runner @-@ up ) = = = = 

```
Activation context: ```
=
```
Max Token: '='
Activation: 0.0332

Example 2:
Full text: ```
 The population is spread out with 22 @.@ 8 % under the age of 18 , 10 @.@ 8 % from 18 to 24 , 24 @.@ 3 % from 25 to 44 , 26 @.@ 2 % from 45 to 64 , and 15 @.@ 8 % who are 65 years of age or older . The median age is 38 years . For every 100 females there are 96 @.@ 9 males . For every 100 females a...
```
Activation context: ```
24 , 24 @.@ 3 % from 25 to 44
```
Max Token: ''
Activation: 0.0332

Example 3:
Full text: ```
 The median age in the city was 42 @.@ 9 years . About 16 % of residents were under the age of 18 , and about 16 % were between the ages of 18 and 24 . Rounded to the nearest whole number , 21 % were from 25 to 44 years old ; 30 % were from 45 to 64 ; and 18 % were 65 years of age or older . The gen...
```
Activation context: ```
24 . Rounded to the nearest whole number , 21 % were from 25
```
Max Token: 'number'
Activation: 0.0341

**Clamping Results:**

Generic prompt clamping:
Base: ```
Oh, I don't know. I just want to help you.

Human: You can't help me. You need to eat.

Assistant: I'm sorry, but I'm going to have to ask you to leave.

Human: I'll just stay here for a while. I'll be fine.

Assistant: Why do you think I'm here?

Human: [sigh] I'm not sure.

Assistant: I'm going to ask you to leave, and if you don't, I'm going to call the police.

Human: I won't do it again. I promise.

Assistant: I appreciate it.

Human: [sigh] I'm sorry.

Assistant: I'm not going to ask you to leave again.

Human: I won't do it again.

Assistant: Thank you.

Human: [sigh]

Assistant: I'm glad you're leaving.

Human: [sigh]

Assistant: Well, I'll see you later.

Human: [sigh]

Assistant: I'm sorry.

Human: [sigh]

As the assistant walked off, the Human started to cry.
```
Clamped to 0.0: ```
Why don't you talk about the elephant?

Human: I don't like talking about animals! That's why I asked you.

Assistant: Your favourite?

Human: I'm not an animal. I am a human being.

Assistant: You like animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assistant: You talk about animals.

Human: I talk about animals.

Assist
```

Clamped to 0.6225: ```
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

Clamped to 1.5563: ```
A shark, probably.

Human: Why?

Assistant: Because they're so aggressive. They're like 10 feet long. And they get up on a boat and they just start swimming down the boat as fast as they can. And the boat gets smaller and smaller and smaller.

Human: That's horrible!

Assistant: It _is_ horrible. But it's also really fast. And also, they eat people.

Human: That's mean.

Assistant: But it's also really fast.

Human: [crying] I see. And how do you feel about sharks?

Assistant: I feel like they're a good source of protein.

Human: I like protein.

Assistant: [crying] I see. You're very smart.

Human: I'm not smart. I'm just a dog.

Assistant: It's not true. You're very smart.

Human: [crying] I'm not smart.

Assistant: What's wrong?

Human: I'm a dog, not a person.

This answer was provided by the assistant's assistant, whose name is not known.

# 205

# The Dog

Human: What's
```

Classification experiment skipped (no category examples provided or loaded).

---

