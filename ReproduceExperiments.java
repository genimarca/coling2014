import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * dTo reproduce our experiments, please use Weka 3.7.9 + LibSVM 1.0.5
 * @author fermin
 */
public class ReproduceExperiments {

        public static void main(String[] args) throws Exception{
            String baseDir ="./";
            double[] combined_COAH = crossValidation(baseDir+"combined_COAH.arff");
            double[] combined_MC = crossValidation(baseDir+"combined_MC.arff");
            double[] iSOL_COAH = crossValidation(baseDir+"iSOL_COAH.arff");
            double[] iSOL_MC = crossValidation(baseDir+"iSOL_MC.arff");
            double[] sentiCon_COAH = crossValidation(baseDir+"SentiCon_COAH.arff");
            double[] sentiCon_MC = crossValidation(baseDir+"SentiCon_MC.arff");

            showResults(iSOL_COAH,"iSOL COAH");
            showResults(iSOL_MC,"iSOL MC");
            showResults(sentiCon_COAH,"SentiCon COAH");
            showResults(sentiCon_MC,"SentiCon MC");
            showResults(combined_COAH,"Combined COAH");
            showResults(combined_MC,"Combined MC");
        }

        private static void showResults(double[] results, String experiment){
            System.out.println(experiment+"\tacc.\tprec.\trec.\tf1");
            System.out.println("\t\t"+format(results[0])+"\t"+format(results[1])+"\t"+format(results[2])+"\t"+format(results[3]));
        }

        private static double[] crossValidation(String arffFilename) throws Exception{
            DataSource source = new DataSource(arffFilename);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            String[] options = weka.core.Utils.splitOptions("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -Z -seed 1");
            LibSVM svm = new LibSVM();         // new instance of tree
            svm.setOptions(options);     // set the options
            //svm.buildClassifier(data);   // build classifier

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(svm, data, 10, new Random(1));
            double accuracy = eval.correct()/(eval.correct()+eval.incorrect());
            double precision = eval.weightedPrecision();
            double recall = eval.weightedRecall();
            double f1 = eval.weightedFMeasure();

            return new double[]{accuracy,precision,recall,f1};
        }

    private static double format(double d) {
        return ((int)(d*10000))/10000.0;
    }
}
