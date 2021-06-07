import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.*;

public class preprocess {

    @SuppressWarnings("unchecked")
    public static void main(String[] args) {
        // set up pipeline properties
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,");
        // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        // build pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Read file line by line
        try {
            BufferedReader reader = new BufferedReader(new FileReader(args[0]));
            FileWriter writer = new FileWriter(args[1]);
            String line;
            line = reader.readLine();
            int cnt = 0;
            while (line != null) {
                String arr[] = line.split("\t");
                JSONObject docObj = new JSONObject();
                docObj.put("class", Integer.parseInt(arr[0]));
                JSONArray sentArr = new JSONArray();
                // create a document object
                CoreDocument document = new CoreDocument(arr[1].trim());
                // annnotate the document
                pipeline.annotate(document);
                for (CoreSentence sent : document.sentences()) {
                    JSONObject sentObj = new JSONObject();
                    JSONArray tokenArr = new JSONArray();
                    JSONArray lemmaArr = new JSONArray();
                    for (CoreLabel token: sent.tokens()) {
                        tokenArr.add(token.word());
                        lemmaArr.add(token.lemma());
                    }
                    sentObj.put("tokens", tokenArr);
                    sentObj.put("lemmas", lemmaArr);
                    sentObj.put("parse_tree", sent.constituencyParse().toString());
                    sentArr.add(sentObj);
                }
                docObj.put("sentences", sentArr);
                JSONArray corefClusters = new JSONArray();
                Map<Integer, CorefChain> corefChains = document.corefChains();
                for (CorefChain cc : corefChains.values()) {
                    JSONArray corefArr = new JSONArray();
                    for (CorefChain.CorefMention m: cc.getMentionsInTextualOrder()) {
                        JSONArray mentionArr = new JSONArray();
                        mentionArr.add(m.sentNum-1);      // need -1 since starting from 1
                        mentionArr.add(m.startIndex-1);   // need -1
                        mentionArr.add(m.endIndex-1);     // need -1
                        corefArr.add(mentionArr);
                    }
                    corefClusters.add(corefArr);
                }
                docObj.put("coref_clusters", corefClusters);
                writer.write(docObj.toJSONString() + "\n");
                cnt += 1;
                if (cnt % 500 == 0) {
                    System.out.println(java.time.LocalDate.now().toString() + " "
                        + java.time.LocalTime.now().toString() + ": Processed " + cnt + "lines");
                }
                line = reader.readLine();
            }
            writer.close();
        } catch (IOException e) { e.printStackTrace(); }
    }
}
