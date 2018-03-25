from helper import dissect

from wordscoring import algorithms as ws

mainText = "In a 2010 interview, David Heyman briefly explained the book-to-film transition. " \
           "He commented on Rowling's involvement in the series, stating that she understood that " \
           "books and films are different and was the best support a producer could have. " \
           "Rowling had overall approval on the scripts, which were viewed and discussed " \
           "by the director and the producers. " \
           "Heyman also said that Kloves was the key voice in the process of adapting the novels " \
           "and that certain aspects from the books needed to have been excluded from the scripts " \
           "due to the filmmakers decision to keep the main focus on Harry's journey as a character, " \
           "which would ultimately give the films a defined structure. " \
           "Heyman mentioned that some fans don't " \
           "necessarily understand the adaptation process and that the filmmakers would " \
           "have loved to have everything from the books in the films but noted that it " \
           "was not possible since they had neither time nor cinematic structure to do so." \
           " He finished by saying that adapting a novel to the screen is a really considered process. " \
           "Because the films were being made as the novels were being published, the filmmakers had no " \
           "idea of the story's outcome until the release of the final novel in 2007. " \
           "Kloves spoke of his relationship with Rowling when adapting the novels by saying, The thing is about Jo, " \
           "which is remarkable for someone who had no experience with the filmmaking process, was her intuition. " \
           "We had a conversation the very first day I met her where she said, " \
           "'I know the movies can't be the books... because I know what's coming and it's " \
           "impossible to fully dramatise on screen what I'm going to write. " \
           "Kloves also said, I don't know what compelled me to say this to Rowling, " \
           "but I said, I have just got to warn you my favourite character is not Harry. " \
           "My favourite character is Hermione. " \
           "And I think for some weird reason, " \
           "from that moment on, she sort of trusted me."


l, sentences = dissect.get_sentences(mainText)
print(l, "------", sentences, '\n')

sent_scores = []

for sent in sentences:
    words_list = dissect.remove_stopwords(sent)
    #print(words_list)

    score = ws.word_frequency(words_list) + \
            ws.tf_idf(l, words_list, sentences) + \
            ws.upper_case(words_list) + \
            ws.proper_noun(words_list) + \
            ws.lexical_similarity(words_list) + \
            ws.n_gram(words_list)

    sent_scores.append(score)

print(sent_scores, '\n')

dissect.print_summarized_sentence(sent_scores, sentences)

