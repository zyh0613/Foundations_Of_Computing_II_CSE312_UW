package SimpleEmailSpamFilter;

import java.io.*;
import java.util.*;

public class FindBestSmooth {
    private static final String TRAINSPAM = "./src/data/train/spam";
    private static final String TRAINHAM = "./src/data/train/ham";
    private static final String TEST = "./src/data/test/";

    private static Map<String, Double> PwOverSpam;   // stores P(w|S) for each word in all spam emails.
    private static Map<String, Double> PwOverHam;    // stores P(w|H) for each word in all ham emails.
    private static final double numOfSpam = 1340.0;  // total number of spam emails
    private static final double numOfHam = 3317.0;   // total number of ham emails
    private static final double PofSpam = numOfSpam / (numOfHam + numOfSpam);    // P(S) = # of spam emails / total emails
    private static final double PofHam = 1 - PofSpam;    // P(H) =  1 - P(S)

    private static double smooth;  // the Laplace smoothing number

    public static void main(String[] args) {
        for (smooth = 1.0; smooth >= 1.0 / 99999999999999999999999999.0; smooth /= 2.0) {

            PwOverHam = calculatePwOver(TRAINHAM);
            PwOverSpam = calculatePwOver(TRAINSPAM);

            testUnlabelledEmails(TEST);
        }
    }

    /**
     * Label the unlabelled emails to either spam or ham using the "Naive" Bayse Algorithm.
     * Results would be printed to STDOUT.
     *
     * We check an email by:
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
     * To do this, we first assume that the words in the email are conditionally independent of each other, given that we know
     * whether or not the email is spam.
     *
     * We assume it because our "Naive" Bayse Algorithm works based on it. Under this assumption, P(x1,x2...xn,Spam) can be easily
     * computed by P(x1|S)P(x2|S)...P(xn|S), which is way more easier than the situation that without the assumption, resulting in
     * P(x1,x2...xn,Spam) = P(x1|x2,...,xn,Spam)...P(xn-1|xn,Spam)P(xn|Spam)P(Spam)
     *
     * Note that the assumption itself is not true in real life.
     *
     *
     *
     * @param directoryPath label all emails in this directory
     */
    private static void testUnlabelledEmails(String directoryPath) {
        try {
            PrintStream out = new PrintStream(new FileOutputStream("./FindBestSmooth/smooth" + smooth + ".txt"));
            System.setOut(out);
            System.setOut(out);
            File dir = new File(directoryPath);
            File[] directoryListing = dir.listFiles();
            if (directoryListing != null) {
                for (File file : directoryListing) {
                    out.print(file.toPath().getFileName() + " ");
                    if (isSpam(file)) {
                        out.println("spam");
                    } else {
                        out.println("ham");
                    }
                }
            } else {
                System.err.println("Error occured!!!!!");
                System.exit(1);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static Map<String, Double> calculatePwOver(String path) {
        Map<String, Double> ret = new HashMap<>();

        File dir = new File(path);
        File[] directoryListing = dir.listFiles();
        if (directoryListing == null) {
            System.err.println("Path not found.");
            System.exit(1);
        }
        double numFiles = directoryListing.length;

        for (File file : directoryListing) {
            // Do something with ham file
            try {
                Set<String> token = tokenSet(file);
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

        for (String word : ret.keySet()) {
            ret.put(word, (1.0 * (ret.get(word) + smooth)) / ((numFiles + smooth * 2.0) * 1.0));
        }
        return ret;
    }

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
}
