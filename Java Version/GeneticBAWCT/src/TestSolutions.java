import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class TestSolutions {

	public static void main(String[] args) throws FileNotFoundException {
		int numScenario = 20;
		for (int nA = 50; nA <= 250; nA += 50) {
			for (int nB = nA; nB <= Math.min(251, nA + 50); nB += 50) {
				Scanner scanner_input = new Scanner(new File("src\\Dataset\\" + nA + "_" + nB + ".txt"));
				Scanner scanner_solution = new Scanner(new File("src\\Solutions\\" + nA + "_" + nB + ".txt"));

				for (int i = 0; i < numScenario; i++) {

					String instance = scanner_input.nextLine();
					String[] p_w = instance.split(";");
					String[] schedule = scanner_solution.nextLine().split(" ");

					int p[] = new int[nA + nB];
					int w[] = new int[nA + nB];
					int schedule_pos[] = new int[nA + nB];

					String[] processingTimeList = p_w[0].split(",");
					String[] weightsList = p_w[1].split(",");

					for (int index = 0; index < nA + nB; index++) {
						p[index] = Integer.valueOf(processingTimeList[index]);
						w[index] = Integer.valueOf(weightsList[index]);
						schedule_pos[index] = Integer.valueOf(schedule[index]);
					}

					double sumA = 0, sumB = 0, cumulative = 0;
					for (int index = 0; index < nA + nB; index++) {
						cumulative += p[schedule_pos[index]];
						if (schedule_pos[index] < nA)
							sumA += cumulative * w[schedule_pos[index]];
						else
							sumB += cumulative * w[schedule_pos[index]];

					}
					System.out.println(nA + " " + nB + " " + sumA / nA + " " + sumB / nB);
					if (sumA / nA - sumB / nB != 0)
						throw new RuntimeException("error");
				}
			}
		}
	}

}
