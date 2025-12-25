---
title: "Emergent Communication: An Introduction"
date: 2019-08-01
description: "An accessible introduction to emergent communication and why it matters."
tags: ["emergent communication", "introduction", "multi-agent"]
---

For the past two years, I have been fascinated by the field of emergent communication; I even made it the topic of my master's thesis. This year, I am co-organizing the event that sparked my interest: [The Workshop on Emergent Communication at NeurIPS 2019](https://sites.google.com/view/emecom2019). To give others a look into the field (and also to make my thesis writing more fun), I will be writing a series of blog posts on emergent communication that are meant to be accessible and interesting. I'm hoping to communicate some of the magic that has captivated me.

This is the first post in the series and my goal is explain the basic idea and give a couple different exciting motivations for studying it.

*Update: this post was also translated to [Russian](https://habr.com/ru/post/496830/) by Anton Alexeyev*

## What is Emergent Communication?

Emergent Communication (EC) is the study of learning "communication protocols in order to share information that is needed to solve [some] task" (Foerster et al, 2016), between two or more agents in an environment. {{< sidenote >}}This is meant to be a simplified description and overlooks some related fields and their history (e.g. Signalling). Careful readers will notice that this description is also vague in regards to _where_ the communication takes place {{< /sidenote >}} In the modern incarnation, EC is within deep learning, specifically a subfield of deep multi-agent reinforcement learning (MARL) but also having ties to language.

The common setup has agents in an environment with the ability to send communication on some linguistic channel that other agents can see. Agents are usually initialized without any prior agreement on what that communication should look like and learn with RL to agree on a protocol. That protocol allows them to coordinate with each other and transfer information effectively. This description is quite abstract, so to make it clear, let's look at one of the simplest setups.

### Sender Receiver Games

**Sender-Receiver games** are a basic game between two players: a sender and a receiver{{< sidenote >}}There are many names for this type of game from "signalling games" ([Skryms, 2010](https://www.oxfordscholarship.com/view/10.1093/acprof:oso/9780199580828.001.0001/acprof-9780199580828)) to "referential games" ([Lazaridou et al, 2017](https://arxiv.org/abs/1612.07182)) and others.{{< /sidenote >}}. The sender has some information that the receiver doesn't know, and the goal of the game is for the sender to communicate that information and the receiver to understand it.

![Basic Sender Receiver Game](/images/emecom.png)

For example, the sender can be given one of three shapes: triangle, square, pentagon. All the receiver knows is that the answer is one of the shapes but doesn't know which one. We choose a vocabulary for the sender to use and the sender then creates message to send to the receiver. The receiver must use that message to guess which of the shapes was given to the sender and then both agents get a reward based on whether the receiver's answer is correct. Through iterations of training, they can learn to agree which messages correspond to which numbers. If, for example, a message is a single token from a vocabulary of three possible tokens, you can see how the two agents could learn to map each shape to a token[3]: 0 = square, 1 = pentagon, 2 = triangle.{{< sidenote >}} This mapping of objects in the environment to linguistic symbols is known as "grounding" in machine learning. For a good discussion on this term, see [Chris Manning's talk at VIGiL @ NeurIPS 2018](https://bluejeans.com/playback/s/jftkhICjhUnEbcglGD4qWWpHsvunBNISIZNdGdUo2AD7vD9nAq5aI2yXus70immP) {{< /sidenote >}}

What is important to note here is that there is usually no single "correct" protocol that we are trying to learn; square could just as well be mapped to token 1 or 2 or even a distribution over tokens that the receiver guesses with some probability. EC is not about supervised learning of a specific protocol but about learning any protocol that will allow us to solve the game. If the game is such that we don't receive any reward for guessing square correctly, then we may not care about its mapping at all. In short, EC is not just about communicating all the information, but about communicating information that's useful for that game. Since usefulness is defined by the game and the rewards, the key idea is that we **learn a communication protocol to optimize playing the game**, whatever it may be.

### More Complex Games
EC isn't restricted to such simple games, though. A natural extension is using more complex data such as images [(Lazaridou et al, 2017)](https://arxiv.org/abs/1612.07182) or a complicated game environment [(Resnick et al, 2018)](https://www.pommerman.com/). Our messages can be variable length [(Havrylov and Titov, 2017)](https://arxiv.org/abs/1705.11192) and we can have multiple rounds of back-and-forth communication [(Das, Kottur et al 2017)](https://arxiv.org/abs/1703.06585). We can make both agents able to send and receive and therefore look at 3 or more agents all talking and acting simultaneously [(Foerster et al, 2016)](https://arxiv.org/abs/1605.06676). Even the cooperative nature of the game is not necessary and we can look at large-scale games with many players and competing preferences [(Leibo et al, 2017)](https://arxiv.org/abs/1702.03037).

It is important, though, to be careful in measuring communication as our environments get more complex. One reason is that agents can be implicitly communicating through an action space (e.g. running towards a target instead of saying "I'm going towards the target"). Another issue, pointed out by [Lowe et al (2019)](https://arxiv.org/abs/1903.05168), is that agents can essentially be just saying what they're doing without adding any extra information (e.g. "I am at the target" while the agent is visibly at the target) and certain metric can mistake this for informative communication. Lowe et al suggest a simple way to measure informative communication is just to look at the reward: if the game with a communication channel allows agents to get higher reward than the exact same game without a communication channel, we can be certain that informative communication is occuring. {{<sidenote>}}This is true for _cooperative_ games but not so clear in competitive games. See [our paper](https://arxiv.org/abs/2101.10276) for a discussion on communication vs manipulation (aka cue-reading) {{</sidenote>}}


## Why Study Emergent Communication?

So now we've described what EC is, the question is why are we doing it in the first place? There is no single answer and the motivation depends on the exact research question studied but it seems to me there are a couple schools of thought that most papers fall in to.

### Modelling Human Language Emergence
Clearly RL agents and environments cannot encompass all the complexity of humans learning and the world we live in but there is still value in simplified models. EC can be used as a model of "how-possibly" human language emerged. Using simplified versions of real scenarios, we can run empirical tests whether certain pressures can cause fundmental parts of human language to emerge.{{<sidenote>}}This is bolstered by reinforcement learning being not just a powerful search method but also having a connection to the actual way humans learn from experience [(Sutton & Barto, 1998)](http://incompleteideas.net/book/the-book.html){{</sidenote>}}

A hot topic in recent years is how to emerge language that is [compositional](https://plato.stanford.edu/entries/compositionality/) with many EC papers looking at different hypotheses. Achieving compositionality would not just be an interesting scientific discovery but could also be essential to research in language understanding and specifically the problem of systematically generalizing outside the training distribution [(Lake and Baroni, 2018)](https://arxiv.org/abs/1711.00350). In this way, research in reproducing facets of human language can still be applicable to engineering and linguistics outside of the pure scientific interest.

### Learning Better Protocols
But EC does not necessarily need to have connections to human language, the basic setup is computer agents communicating. The modern world is already made up of networks of computers communicating using various protocols from TCP/IP to Bluetooth. The difference is that many existing protocols have never been computationally optimized for their use cases, they are just rule sets or fixed protocols.

This approach makes the bet that learned protocols will be more efficient, more resilient, and better suited to their tasks compared to fixed protocols. For example, learned protocols could implicitly account for the distribution of messages and allow for more efficient messages on average [(Kraska et al, 2018)](https://arxiv.org/abs/1712.01208). Even more exciting, protocols are learned when optimizing a loss function which means that any specific requirements or needs can be incorporated into the loss function and optimized.

These ideas can be combined with the possibility of learning a protocol together with a policy. Self-driving cars are quickly becoming a possibility but should coodrindation between them be limited to the same communication human drivers have? It seems reasonable that self-driving cars could also learn to communicate amongst themselves to better coordinate their actions: from warning cars far behind them of a crash to letting other drivers know their route. The main point is that communication protocols between machines don't need to be limited to the rule sets humans can come up with.

### Modelling and Coordinating with Other Agents

One of the main rules of telling a good joke is knowing your audience. Similarly, good communication requires understanding and modelling the agents you are interacting with. A good sender must understand the difference between what they know themselves and what their receiver knows, then from that difference extract the most essential pieces of information the receiver should get in that moment. So the most effective communication requires opponent modelling as well as good contextual understanding.

This is clear in games like Hanabi [(Bard et al, 2019)](https://arxiv.org/abs/1902.00506) that have implicit communication but it is even more important for explicit strategic communication in competitive games. By using communicative games as a test bed, we can look to improve our communication by improving our opponent-modelling.


### Bottom-Up Natural Language

The final view is something of a moonshot steeped in philosophy: EC as natural language understanding/generation. If we follow Wittgenstein's [Philosophical Investigations (1953)](https://en.wikipedia.org/wiki/Philosophical_Investigations), it isn't just our protocols that get their meaning from use in an environment but human natural language as well. We can consider regular language understanding to be "top-down": looking at text in context and trying to derive the meanings. In contrast, EC seeks to learn "bottom-up": if we learn a language in an environment resembling the real world, our emerged language could be equivalent to natural language.

One idea is that if humans and agents are given similar environments, then an emergent language learned in the environment should be mappable to the language humans use in it. If we manage to learn that mapping, then together with the emergent language we should have a system that understands words by grounding them in their true meaning: how they are used. This could be the right approach to language understanding and potentially more effective than current approaches that understand words by reading text.

## Conclusion

I've introduced the basics of emergent communication and hopefully given a taste of some of the possible research directions. My explanations are not meant to be exhaustive but illustrative to illumate the field. And my criticisms are not meant to prescriptive, my greatest hope is that research in the field (including my own) overcomes the skepticism it faces. Emergent communication is a nascent field with many interesting directions, but many of the ideas here are not new. It is important to also see it in the context of previous work in fields such as evolutionary game theory, anthropology, and information theory.

### Acknowledgements
Thank you to my friend Christine Xu for feedback and editing, my coauthor [Travis Lacroix](https://travislacroix.github.io/) and my workshop co-organizers for read-throughs and support.

### Citation

```
@misc{noukhovitch2019emergentintro
    author = {Michael Noukhovitch},
    title = {Emergent Communication: An Introduction},
    year = {2019},
    url = {mnoukhov.github.io/posts/emergent-communication-introduction}
}
```
