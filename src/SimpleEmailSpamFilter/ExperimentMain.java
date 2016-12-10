package SimpleEmailSpamFilter;

import java.io.*;
import java.util.*;

public class ExperimentMain {
    private static final String TRAINSPAM = "./src/data/train/spam";
    private static final String TRAINHAM = "./src/data/train/ham";
    private static final String TEST = "./src/data/test/";

    private static Map<String, Double> PwOverSpam;   // stores P(w|S) for each word in all spam emails.
    private static Map<String, Double> PwOverHam;    // stores P(w|H) for each word in all ham emails.
    private static final double numOfSpam = 1340.0;  // total number of spam emails
    private static final double numOfHam = 3317.0;   // total number of ham emails
    private static final double PofSpam = numOfSpam / (numOfHam + numOfSpam);    // P(S) = # of spam emails / total emails
    private static final double PofHam = 1 - PofSpam;    // P(H) =  1 - P(S)

    private static double smooth = 1.0;  // the Laplace smoothing number

    public static void main(String[] args) {
        // initialize two maps
        PwOverHam = calculatePwOver(TRAINHAM);
        PwOverSpam = calculatePwOver(TRAINSPAM);

        // get the words with highest and lowest ratio P(w|S) / P(w|H) respectively.
        getAndPrintRatio();

        // label the unlabelled emails
        //testUnlabelledEmails(TEST);
    }

    /*
     * Problem 1:
     *
     * (a) We check an email by:
     *
     * P(S|x1,x2....xn) > 0.5, where x1,...xn are all words in the email; else it is considered as a ham.
     *
     *
     * According to the Bayse Algorithm, and the Law of total Probability, we can calculate P(S|x1,x2...xn) by
     *
     *                                             P(x1|S)P(x2|S)...P(xn|S)
     *          P(S|x1,x2...xn) = --------------------------------------------------------------
     *                              P(S)P(x1|S)P(x2|S)...P(xn|S) + P(H)P(x1|H)P(x2|H)...P(xn|H)
     *
     * To do this, we first assume that the words in the email are conditionally independent of each other, given that
     * we know whether or not the email is spam.
     *
     * We assume it because our "Naive" Bayse Algorithm works based on it. Under this assumption, P(x1,x2...xn,Spam)
     * can be easily computed by P(x1|S)P(x2|S)...P(xn|S), which is way more easier than the situation that without the
     * assumption, resulting in P(x1,x2...xn,Spam) = P(x1|x2,...,xn,Spam)...P(xn-1|xn,Spam)P(xn|Spam)P(Spam)
     *
     * (b) Note that the assumption itself is not true in real life. For example, the words "San Francisco" is more
     * likely appear together and they are not independent; that is, P(San)*P(Francisco) should be less than
     * P(San Francisco) in real life.
     *
     * (c) No. Here is a counter example. According to the problem:
     *          Let Z be an event and P(Z) = 0.5, P(Zc) = 0.5 (Zc means the complementary of Z)
     *          and give let X,Y be a marginal distribution that
     *          Z:                               Zc:
     *              X\Y | 0 | 1 |                   X\Y | 0 | 1 |
     *             ------------------              ------------------
     *               0  |0.2|0.1|0.3                 0  |0.7|0.1|0.8
     *             ------------------              ------------------
     *               1  |0.4|0.3|0.7                 1  |0.2|0.0|0.2
     *             ------------------              ------------------
     *                  |0.6|0.4| 1                     |0.9|0.1| 1
     *
     *          In this case, P(X=0|Z)*P(Y=0) = 0.3 * (0.5 * 0.6 + 0.5 * 0.9) = 0.225, 
     *                  which is equal to P( (X=0|Z) & Y=0 ) = 0.225, and they are independent;
     *             similarly, P(X=0|Zc)*P(Y=0) = 0.9 * (0.5 * 0.6 + 0.5 * 0.9) = 0.675, 
     *                  which is equal to P( (X=0|Z) & Y=0 ) = 0.675, and they are independent;
     *             however, P(X=0) * P(Y=0) = (0.3 * 0.5 + 0.8 * 0.5) * (0.6 * 0.5 + 0.7 * 0.5) = 0.55 * 0.65 = 0.3575,
     *                  which is not equal to P(X=0 & Y=0) = 0.2 * 0.5 + 0.7 * 0.5 = 0.45
     *             so X=0 is not independent to Y=0
     *
     */

    /**
     * Label the unlabelled emails to either spam or ham using the "Naive" Bayse Algorithm.
     * Results would be printed to STDOUT.
     *
     * @param directoryPath label all emails in this directory
     */
    public static void testUnlabelledEmails(String directoryPath) {
        File dir = new File(directoryPath);
        File[] directoryListing = dir.listFiles();
        if (directoryListing != null) {
            for (File file : directoryListing) {
                System.out.print(file.toPath().getFileName() + " ");
                if (isSpam(file)) {
                    System.out.println("spam");
                } else {
                    System.out.println("ham");
                }
            }
        } else {
            System.err.println("Error occured!!!!!");
            System.exit(1);
        }
    }

    // calculate the P(w|S) or P(w|H) for words in all emails in a given directory path.
    private static Map<String, Double> calculatePwOver(String path) {
        Map<String, Double> ret = new HashMap<>();

        File dir = new File(path);
        File[] directoryListing = dir.listFiles();  // all files
        if (directoryListing == null) {
            System.err.println("Path not found.");
            System.exit(1);
        }
        double numFiles = directoryListing.length;

        for (File file : directoryListing) {
            // Do something with ham file
            try {
                Set<String> token = tokenSet(file);

                // count and save the number of occurrences each word (no duplicate) in all emails
                for (String word : token) {
                    if (ret.containsKey(word)) {
                        ret.put(word, ret.get(word) + 1);
                    } else {
                        ret.put(word, 1.0);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        // count and save the P(w|S) or P(w|H) for each word
        for (String word : ret.keySet()) {
            ret.put(word, (1.0 * (ret.get(word) + smooth)) / ((numFiles + smooth * 2.0) * 1.0));
        }
        return ret;
    }

    // get all words in an email file with no duplicate and ignore the "Subject"
    private static HashSet<String> tokenSet(File file) throws IOException {
        HashSet<String> tokens = new HashSet<>();
        Scanner filescan = new Scanner(file);
        filescan.next(); //Ignoring "Subject"
        while(filescan.hasNext()) {
            String word = filescan.next();
            tokens.add(word);
        }
        filescan.close();
        return tokens;
    }

    // check if an email file is spam. The main algorithm has mentioned before.
    private static boolean isSpam(File file) {
        double summationOfLogPofAllWordsGivenS = 0.0;
        double summationOfLogPofAllWordsGivenH = 0.0;
        double logPofSpam = Math.log(PofSpam);
        double logPofHam = Math.log(PofHam);
        try {
            Set<String> token = tokenSet(file);
            for (String word : token) {
                if (PwOverSpam.containsKey(word)) {
                    summationOfLogPofAllWordsGivenS = summationOfLogPofAllWordsGivenS + Math.log(PwOverSpam.get(word));
                } else {
                    summationOfLogPofAllWordsGivenS += Math.log(smooth / (numOfSpam + smooth * 2));
                }
                if (PwOverHam.containsKey(word)) {
                    summationOfLogPofAllWordsGivenH += Math.log(PwOverHam.get(word));
                } else {
                    summationOfLogPofAllWordsGivenH += Math.log(smooth / (numOfHam + smooth * 2));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return logPofSpam + summationOfLogPofAllWordsGivenS > logPofHam + summationOfLogPofAllWordsGivenH;
    }

    // calculate the ratio of all words and find the max one and the min one
    private static void getAndPrintRatio() {
        Map<String, Double> map = PwOverSpam;
        Map<String, Double> map2 = PwOverHam;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        String w = "";
        String w2 = "";
        for (String word : map.keySet()) {
            if (map2.containsKey(word)) {
                if ((1.0 * map.get(word)) / (1.0 * map2.get(word)) < min) {
                    w = word;
                    min = (1.0 * map.get(word)) / (1.0 * map2.get(word));
                }
                if ((1.0 * map.get(word)) / (1.0 * map2.get(word)) > max) {
                    w2 = word;
                    max = (1.0 * map.get(word)) / (1.0 * map2.get(word));
                }
            } else {
                if ((1.0 * map.get(word)) / (1.0 / (numOfHam + 2.0)) < min) {
                    w = word;
                    min = (1.0 * map.get(word)) / (1.0 / (numOfHam + 2.0));
                }
                if ((1.0 * map.get(word)) / (1.0 / (numOfHam + 2.0)) > max) {
                    w2 = word;
                    max = (1.0 * map.get(word)) / (1.0 / (numOfHam + 2.0));
                }
            }
        }
        for (String word : map2.keySet()) {
            if (map.containsKey(word)) {
                if ((1.0 * map.get(word)) / (1.0 * map2.get(word)) < min) {
                    w = word;
                    min = (1.0 * map.get(word)) / (1.0 * map2.get(word));
                }
                if ((1.0 * map.get(word)) / (1.0 * map2.get(word)) > max) {
                    w2 = word;
                    max = (1.0 * map.get(word)) / (1.0 * map2.get(word));
                }
            } else {
                if (1.0 / (numOfSpam + 2.0) / (1.0 * map2.get(word)) < min) {
                    w = word;
                    min = 1.0 / (numOfSpam + 2.0) / (1.0 * map2.get(word));
                }
                if (1.0 / (numOfSpam + 2.0) / (1.0 * map2.get(word)) > max) {
                    w2 = word;
                    max = 1.0 / (numOfSpam + 2.0) / (1.0 * map2.get(word));
                }
            }
        }
        System.out.println("min: word: " + w + " ratio:" + min);
        System.out.println("max: word: " + w2 + " ratio:" + max);
    }
}