--% Machine Learning Online Class
--  Exercise 6 | Spam Classification with SVMs
--
--  Instructions
--  ------------
-- 
--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:
--
--     gaussianKernel.m
--     dataset3Params.m
--     processEmail.m
--     emailFeatures.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--

package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method6"
local plot = require"gnuplot"
local optim = require"optim"
local nn = require"nn"

local string_format = string.format

--% Initialization
misc.clear_screen()

--% ==================== Part 1: Email Preprocessing ====================
--  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
--  to convert each email into a vector of features. In this part, you will
--  implement the preprocessing steps for each email. You should
--  complete the code in processEmail.m to produce a word indices vector
--  for a given email.

misc.printf('\nPreprocessing sample email (emailSample1.txt)\n');

-- Extract Features
local f = io.open("emailSample1.txt", "r")
local file_contents = f:read("*a")
f:close()
local word_indices = method.process_email(file_contents)

-- Print Stats
misc.printf('Word Indices: \n');
misc.printf(table.concat(word_indices, " "))
misc.printf('\n\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% ==================== Part 2: Feature Extraction ====================
--  Now, you will convert each email into a vector of features in R^n. 
--  You should complete the code in emailFeatures.m to produce a feature
--  vector for a given email.

misc.printf('\nExtracting features from sample email (emailSample1.txt)\n');

-- Extract Features
local features = method.email_features(word_indices)

-- Print Stats
misc.printf('Length of feature vector: %d\n', features:numel());
misc.printf('Number of non-zero entries: %d\n', torch.gt(features, 0):sum());

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% =========== Part 3: Train Linear SVM for Spam Classification ========
--  In this section, you will train a linear classifier to determine if an
--  email is Spam or Not-Spam.

-- Load the Spam Email dataset
-- You will have X, y in your environment
local load_rlt = loader.load_from_mat('spamTrain.mat');
local X = load_rlt.X
local y = load_rlt.y

misc.printf('\nTraining Linear SVM (Spam Classification)\n')
misc.printf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
local model = method.svm_train(X, y, C, 0, 0.0001, 20)
local p, accuracy = method.svm_predict(model, X, y)

misc.printf('Training Accuracy: %f\n', 
    torch.eq(p, y):double():mean() * 100);

--% =================== Part 4: Test Spam Classification ================
--  After training the classifier, we can evaluate it on a test set. We have
--  included a test set in spamTest.mat

-- Load the test dataset
-- You will have Xtest, ytest in your environment
load('spamTest.mat');
local load_rlt = loader.load_from_mat('spamTest.mat');
local Xtest = load_rlt.Xtest
local ytest = load_rlt.ytest

misc.printf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = method.svm_predict(model, Xtest, ytest);

misc.printf('Test Accuracy: %f\n',
    torch.eq(p, ytest):double():mean() * 100);
misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% ================= Part 5: Top Predictors of Spam ====================
--  Since the model we are training is a linear SVM, we can inspect the
--  weights learned by the model to understand better how it is determining
--  whether an email is spam or not. The following code finds the words with
--  the highest weights in the classifier. Informally, the classifier
--  'thinks' that these words are the most likely indicators of spam.
--

-- Sort the weights and obtin the vocabulary list
local _, vocab_list = method.get_vocab_list(true)
local idx_weight = method.svm_predict_weight(model, vocab_list)

misc.printf('\nTop predictors of spam: \n');
for i = 1, 15, 1 do
    local node = idx_weight[i]
    local idx = node[1]
    local weight = node[2]
    misc.printf(' %s (%f) \n', vocab_list[idx], weight);
end

misc.printf('\n\n');
misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

--% =================== Part 6: Try Your Own Emails =====================
--  Now that you've trained the spam classifier, you can use it on your own
--  emails! In the starter code, we have included spamSample1.txt,
--  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
--  The following code reads in one of these emails and then uses your 
--  learned SVM classifier to determine whether the email is Spam or 
--  Not Spam

-- Set the file to be read in (change this to spamSample2.txt,
-- emailSample1.txt or emailSample2.txt to see different predictions on
-- different emails types). Try your own emails as well!
local filenames = {'spamSample1.txt', 'spamSample2.txt', 'emailSample1.txt', 'emailSample2.txt'}
for _, filename in ipairs(filenames) do
    misc.printf("For file content from: %s", filename)

    local f = io.open(filename, "r")
    local file_contents = f:read("*a")
    f:close()
    local word_indices = method.process_email(file_contents)
    local x = method.email_features(word_indices)
    local p = method.svm_predict(model, x)

    misc.printf('\nProcessed %s\n\nSpam Classification: %s\n', filename, tostring(p));
    misc.printf('(1 indicates spam, 0 indicates not spam)\n\n');
end


