1.What should you do if the two models have different tokenizers?
   We can use force them to use the same tokenizers and then apply the algorithm to the tokens, in the code provided itj ust used 1 tokenizer
   or we can manually create code that convert one to another, but usually it's not advised to use tokenizers that doesn't match the model
2.Do you think contrastive decoding is used in practice?
    I don't think it's used in practice, this is because this takes way too much extra resources. it runs through two different models and then compare them to each other. I feel like if computing can be improved this method will be more practical but it's just way too costly right now to run it.

I used the extra time to set up my computer, it ran into some errors when I forked it and try to pull it to my computer.