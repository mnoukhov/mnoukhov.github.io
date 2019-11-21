# Emergent Communication: The What and The Why

For the past two years, I have been fascinated by the field of emergent communication; I even made it the topic of my master's thesis. This year, I am helping organize the event that sparked my interest: [the emergent communication workshop at NeurIPS](sites.google.com/view/emecom2019). To give others a look into the field (and also to make my thesis writing more fun), I will be writing a series of blog posts on emergent communication that are meant to be accessible and interesting. I'm hoping to communicate some of the magic that has captivated me.

This is the first post in the series and my goal is explain the basic concept and describe some of the different motivations in studying it.

## What is Emergent Communication?

Emergent Communication (EC) is the study of learning a linguistic communication protocol between two or more (usually deep RL) agents in an environment [^1]. In this way, you can think of EC as being within deep learning, specifically a subfield of deep multi-agent reinforcement learning, but also with ties to language. The common setup is of agents in an environment with the ability to send communication on some linguistic channel that other agents can see but initialized without any prior agreement on what that communication should look like. Agents treat communication as an action space and using RL can learn to agree on a protocol that allows them to coordinate with each other and transfer information effectively.

That description is quite abstract, so to make it clear, let's look at the most basic setup

### Sender Receiver Games

__Sender-Receiver games__ are a basic game between two players: a sender and a receiver[^2]. The sender has some information that the receiver doesn't know, and the goal of the game is for the sender to communicate that information and the receiver to understand it.

![Basic Sender Receiver Game](sender-receiver.png)

For example, the sender can be given one of three shape: triangle, square, pentagon and all the receiver knows is that the answer is one of the shapes but doesn't know which one. The sender chooses a message to send to the receiver and the receiver must use that message to guess what the number is. Both agents get a reward based on whether the receiver's answer is correct, and so they can learn over training to agree which messages correspond to which numbers. If, for example, the communication was restricted to a single number chosen between 0-2, you can see how the two agents could learn to map each shape to a symbol[^3]: 0 = square, 1 = pentagon, 2 = triangle.

What is important to note here is that there is usually no single "correct" protocol that we are trying to learn; square could just as well be mapped to 1 or 2 or even stochastically a distribution over the numbers. EC is not about supervised learning of a specific protocol but about learning any protocol that will allow us to solve the game. If the game is such that we don't receive any reward for guessing square correctly, then we may not care about its mapping at all. In short, EC is not just about communicating _all_ the information, but about communicating information _effectively_. Effectiveness is defined by the game and the rewards, so the key idea is that __the communication protocol is learned to optimize playing the game__.

### More Complex Games

We don't need to restrict ourselves to a single fixed sender and receiver. We can look at populations of senders and receiver (CITE), at agents that both send and receive (CITE), and at multi-agent scenarios with many agents all sending and receiving (CITE).

As well, we can have more complex interactions. We can still look at sender-receiver games but have multiple rounds of back-and-forth communication before the sender's action (CITE). We can look at large-scale games with many players and different preferences, allowing them to use communication to coordinate (CITE).

It is important, though, to be careful in measuring communication as our environments get more complex. One reason is that agents can be implicitly communicating through an action space (e.g. running towards a target instead of saying "I'm going towards the target"). Another issue, pointed out by Lowe et al (2019), is that agents can essentially be just saying what they're doing without adding any extra information (e.g. "I am at the target" while the agent is visibly at the target) and certain metric can mistake this for informative communication. Lowe et al suggest a simple way to measure informative communication is just to look at the reward: if the game with a communication channel allows agents to get higher reward than the exact same game without a communication channel, we can be certain that informative communication is occuring.[^4]


## Why Study Emergent Communication?

So now we've described _what_ EC is, the question is _why_ are we doing it in the first place? There is no single answer and the motivation depends on the exact research question studied but it seems to me there are a couple schools of thought that most papers fall in to.

Here, I describe the main approaches I've noticed and arguments for and against each perspective.

### Modelling Human Language Emergence

Clearly RL agents and environments cannot encompass all the complexity of humans learning and the world we live in but there is still value in simplified models. EC can be used as a model of "how-possibly" human language emerged.. Using simplified versions of real scenarios, we can run empirical tests whether certain pressures can cause fundmental parts of human language to emerge [^5].

A hot topic in recent years is how to emerge language that is [compositional](https://plato.stanford.edu/entries/compositionality/) with many EC papers going that route (CITE). Achieving compositionality would not just be an interesting scientific discovery but could also be essential to research in language understanding and specifically the problem of systematically generalizing outside the training distribution (CITE Baroni). In this way, research in reproducing facets of human language can still be applicable to engineering and linguistics outside of the pure scientific interest.

I can see criticism of this approach from two different perspectives: antropologists and engineers. The anthropological criticism is that "how-possibly" is not a good enough model. This perspective would argue that research into human language should look at humans and their actual history, using methods tied closer to reality that modelling. The engineering criticism is completely different, and would claim that there is little practical use in reproducing human language (with some exceptions). This persepective is mostly about seeing AI research as building tools not making contributions to anthropology.

But EC does not necessarily need to have connections to human language

### Learning Better Protocols

The basic setup of EC is _computer agents_ communicating and the modern world is already made up of networks of computers communicating using various protocols from TCP/IP to Bluetooth. The difference is that many existing protocols have never been computationally optimized for their use cases, they are _fixed_ protocols.

This approach makes the bet that learned protocols will be more efficient, more resilient, and better suited to their tasks compared to fixed protocols. For example, learned protocols could implicitly account for the distribution of messages and allow for more efficient messages on average. Even more exciting, protocols are learned when optimizing a loss function which means that any specific requirements or needs can be incorporated into the loss function and optimized.

These ideas can be combined with the possibility of learning a protocol together with a policy. Self-driving cars are quickly becoming a possibility but should coodrindation between them be limited to the same communication human drivers have? It seems reasonable that self-driving cars could also learn to communicate amongst themselves to better coordinate their actions: from warning cars far behind them of a crash to letting other drivers know their route. The main point is that communication protocols between machines don't need to be limited to protocols between humans.

Criticism of this and similar approaches seems to focuses on interpretability and the practical applicability. For one, emergent protocols are inherently less interpretable than fixed, designed protocols and this can contribute to the guarantees we can make and the trust we have. TCP is guaranteed to send all packets until the receiver is satisfied but an emergent protocol may not. Even worse, it may not be clear whether it would or adversarial examples could cause it to fail.

As for the superiority of learned vs fixed, a similar argument has already been made for index structures ([Kraska et al, 2018][Kraksa2018]). Critics have argued that the added complexity of a neural network would eliminate any practical efficiency improvements.

### Modelling Other Agents

One of the main rules of telling a good joke is knowing your audience. Similarly, good communication requires understanding and modelling the agents you are interacting with. A good sender must understand the difference between what they know themselves and what their receiver knows, then from that difference extract the most essential pieces of information the receiver should get in that moment. So the most effective communication requires opponent modelling as well as good contextual understanding.

It's hard to argue that opponent modelling isn't a useful addition to communication, so the critical view is that communication is not necessary to opponent modelling. Games like Hanabi (CITE) have all the elements of opponent modelling but don't require explicit linguistic communication, so perhaps communication is just adding complexity.

### Bottom-Up Language Understanding

The final view is something of a moonshot steeped in philosophy: EC as natural language understanding. Based on Wittgenstein's idea of language games, it isn't just our protocols that are learned by optimizing in an environment but human natural language as well. We can consider regular language understanding to be "top-down": looking at text in context and try to derive the meanings. In contrast, this approach seeks to learn "bottom-up": learning an emergent language starting by grounding meanings and then building up a corpus of text.

The idea is that if humans and agents are given similar environments, then an emergent language learned in the environment should be mappable to the language humans use in it. If we manage to learn that mapping, then together with the emergent language we should have a system that understands meanings of words "grounded" in their use in the environment. This language understanding of words in the context of actions and environment should be deeper and more effective than the top-down approach that can only capture words in the context of other words.

The criticism here is also the idea: this is a moonshot. There are a couple assumptions both practical and philosophical and there are no guarantees that they will all pan out.


## Conclusion

I've introduced the basics of emergent communication and hopefully given a taste of some of the possible research directions. My explanations are not meant to be exhaustive but illustrative to illumate the field. And my criticisms are not meant to prescriptive, my greatest hope is that research in the field (including my own) overcomes the skepticism it faces. Emergent communication is a nascent field with many interesting directions, but many of the ideas here are not new. It is important to also see it in the context of previous work in fields such as evolutionary game theory, anthropology, and information theory. Look forward to that context being expanded upon in the next post!

If you've found any errors (even minor) or have commentary please email me `mnoukhov@gmail.com`. I would love to get feedback and improve this for others, and I appreciate all notes! If you'd like to talk, in person, about emergent communication then please come to our workshop at NeurIPS 2019!

### Acknowledgements



## Footnotes

[^1]: This is meant to be a simplified description and overlooks some related fields and their history (e.g. [Signalling]()) but we will expand on those in future posts. Careful readers will notice that this description is also vague in regards to _where_ the communication takes place.

[^2]: There are many names for this type of game from "signalling games" ([Skryms, 2016]()) to "referential games" ([Lazaridou et al, 2016]()) and others.

[^3]: This mapping of objects in the environment to linguistic symbols is known as "grounding" in machine learning. For a good discussion on this term, see [Chris Manning's talk at VIGiL @ NeurIPS 2018]()

[^4]: This is true for _cooperative_ games but not so clear in competitive games. See [our paper]() for a discussion on communication vs manipulation (aka cue-reading)

[^5]:  This is bolstered by reinforcement learning being not just a powerful search method but also having a connection to the actual way humans learn from experience (CITE)


## References

[Foerster et al, 2016]: here

[Kraska2018]: Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean and Neoklis Polyzotis. 2018. The Case for Learned Index Structures. In Proceedings of the 2018 International Conference on Management of Data (SIGMOD '18)


### Cite This

{



