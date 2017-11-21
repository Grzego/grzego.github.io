---
layout: post
title:  "MNIST to handwriting"
date:   2017-11-10 16:00:00 +0100
image:  "/assets/images/mnist-to-handwriting/title.png"
description:  "Training a model for turning MNIST digit images into handwriting."
repository: 
---

Recently on one of my repositories I was ask whether a model for handwriting can be used to generate sequence of numbers. As it turn out, the quality of such handwritten numbers was pretty bad. Although the problem seems quite easy there is no good dataset to train model on. So I decided to try create a model to generate such dataset.

Task is to take an image of digit and label as the input and produce sequence of points representing that digit. Because we are speaking of sequences, natural thing is to try applying Recurrent Neural Network (RNN) [[1]].

<p align="center">
	<img src="{{ "/assets/images/mnist-to-handwriting/title.png" | absolute_url }}"/>
</p>


## But how?

We want sequence of points as the result of the model. My first idea was to unroll RNN for some number of steps, getting at each step a single point and in the result a sequence. But there is one problem, how to train such model?

We only have image and label. Here comes the second idea. We need something able to draw a line. Being able to draw lines from points onto 2D image would allow us to turn sequence of previously generated points into an image of digit. Using _mean square error_ as a loss function we would be able to train RNN. Whole model can be interpreted as an autoencoder, because it tries to reconstruct its input, but we kind of constraint its internal representation (forcing it to be sequence of points).


<p align="center">
  <img src="{{ "/assets/images/mnist-to-handwriting/model.png" | absolute_url }}"/>
</p>


## Drawing lines




---

[(1): Great post about LSTM - http://colah.github.io/posts/2015-08-Understanding-LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs)

[1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs


<!-- You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk]. -->

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
