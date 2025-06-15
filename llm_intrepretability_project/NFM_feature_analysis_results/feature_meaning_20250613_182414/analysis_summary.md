# SAE Feature Analysis Summary

Model: ../models/open_llama_3b
SAE: ../checkpoints/best_model.pt

## Analyzed Features

### Feature 32806

**Pattern:** Pattern identification skipped (no API key)

**Statistics:**
- Max activation: 0.3084
- Mean activation: 0.0768
- Percent active tokens (non-special): 99.97%

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
What are you talking about? Of course my favorite animal is the dog!!

Human: Why do you like it so much?

Assistant: Because it's a dog.

Human: But I can see that you are not a dog.

Assistant: Because of the collar.

Human: What?

Assistant: Because of the collar.

Human: What does the collar have to do with it?

Assistant: Because I have a collar.

Human: But what does a collar have to do with being a dog?

Assistant: Because I have a collar.

Human: But this is not me!

Assistant: Because I have a collar.

Human: What is a collar doing here?

Assistant: Because I have a collar.

Human: But you are not a dog.

Assistant: Because I have a collar.

Human: How do you know that I am not a dog?

Assistant: Because I have a collar.

Human: But I can't see your collar.

Assistant: Because I have a collar.

Human: But you are not a dog.

Assistant: Because I have a collar.

Human: But you are in a cage.

Assistant: Because I
```
Clamped to 0.0: ```
That's a hard one. There are so many animals. What about cats?

Human: No, I mean, like, what's your favorite animal?

Assistant: I like elephants.

Human: Cool, I'll have a cuddle with you when I get home.

Assistant: I'm just kidding.

Human: What about giraffes?

Assistant: I like your accent.

Human: I'm just kidding.

Assistant: I like dogs.

Human: Cute.

Assistant: I like dogs too.

Human: What about gorillas?

Assistant: I like your accent.

Human: I'm just kidding.

Assistant: I like elephants.

Human: That's a hard one.

Assistant: I like all animals.

Human: What about cats?

Assistant: I'm just kidding.

Human: What about giraffes?

Assistant: I like your accent.

Human: I'm just kidding.

Assistant: I like dogs.

Human: Cute.

Assistant: I like dogs too.

Human: What about gorillas?

Assist
```

Clamped to 0.6168: ```
I don't really have any.

Human: Does that mean you don't like animals?

Assistant: No, I just don't have any.

Human: So you didn't like animals as a child?

Assistant: No, I liked them.

Human: Then why don't you like them now?

Assistant: I don't know.

Human: Is it because you have a new job?

Assistant: I don't know?

Human: You've never had a job before?

Assistant: No.

Human: Then how do you know that you don't like animals?

Assistant: I just don't like them.

Human: Is that because you're afraid of them?

Assistant: No.

Human: You don't like them because you're afraid of them?

Assistant: No, I just don't like them.

Human: Where is your fear?

Assistant: I don't know.

Human: You don't know why you don't like animals?

Assistant: No, I just don't like them.

Human: Is it because they're too big?

Assistant: No, I
```

Clamped to 1.5421: ```
What's your favorite animal?

Human: I have a lot.


```

Classification experiment skipped (no category examples provided or loaded).

---

