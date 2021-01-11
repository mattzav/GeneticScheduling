import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

import jxl.Workbook;
import jxl.write.Label;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import jxl.write.WriteException;

public class Main {

	private static final String EXCEL_FILE_LOCATION = "src\\Result\\";
	static WritableWorkbook workBook = null;
	static WritableSheet excelSheet;

	static int[] populationSizeArray = { 10, 25, 50 };
	static double[] crossoverProbabilityArray = { 1, 0.8, 0.6, 0.4, 0.2, 0.1 };
	static double[] mutationProbabilityArray = { 1, 0.8, 0.6, 0.4, 0.2, 0.1 };
	static int[] initMethodArray = { 0, 1, 2 };
	static int[] selectionMethodArray = { 0, 1, 2 };
	static int[] crossoverMethodArray = { 0, 1, 2, 3 };
	static int[] mutationMethodArray = { 0, 1, 2, 3, 4, 5 };

	static Scanner scanner_input;
	static Scanner scanner_output;

	static int populationSize;
	static double mutationProbability, crossoverProbability;
	static boolean optimumFound;
	static double totalFitness;

	static Random r;

	static int nA, nB;
	static double UB;
	static Individual best;
	static int indexBestChildren;
	static Individual[] population;

	static double p[], w[], pwSum, pSum;
	static ArrayList<Integer> _1_n, _1_nA, nA_n;

	// GESTIRE CON FOR NEL MAIN
	static int initParam;
	static int selectionParam;
	static int crossoverParam;
	static int mutationParam;
	//

	public static void main(String[] args) throws FileNotFoundException {
		r = new Random();
		r.setSeed(1);

		for (int populationIndex = 0; populationIndex < populationSizeArray.length; populationIndex++)
			for (int crossoverProbabilityIndex = 0; crossoverProbabilityIndex < crossoverProbabilityArray.length; crossoverProbabilityIndex++)
				for (int mutationProbabilityIndex = 0; mutationProbabilityIndex < mutationProbabilityArray.length; mutationProbabilityIndex++)
					for (int initMethodIndex = 0; initMethodIndex < initMethodArray.length; initMethodIndex++)
						for (int selectionMethodIndex = 0; selectionMethodIndex < selectionMethodArray.length; selectionMethodIndex++)
							for (int crossoverMethodIndex = 0; crossoverMethodIndex < crossoverMethodArray.length; crossoverMethodIndex++)
								for (int mutationMethodIndex = 0; mutationMethodIndex < mutationMethodArray.length; mutationMethodIndex++) {

									initOutParameters(populationIndex, crossoverProbabilityIndex,
											mutationProbabilityIndex, initMethodIndex, selectionMethodIndex,
											crossoverMethodIndex, mutationMethodIndex);

									createExcelFile();

									for (nA = 50; nA <= 250; nA += 50) {
										for (nB = nA; nB <= Math.min(nA + 50, 251); nB += 50) {

											scanner_input = new Scanner(
													new File("src\\Dataset\\" + nA + "_" + nB + ".txt"));
											System.out.println(nA + " " + nB);
											double totalTime = 0, totalObjVal = 0;
											int totalIter = 0;
											/*
											 * popSize = .. mutProb = ..
											 */
											for (int scenario = 0; scenario < 50; scenario++) {
												initParameters();

												long start = System.currentTimeMillis();

												// INIT POPULATION
												if (initParam == 0)
													initRandom();
												else if (initParam == 1)
													initAlternating();
												else if (initParam == 2)
													initBidirectional();
												else
													throw new RuntimeException("init not found");

												// END INIT POPULATION

												double newFitness = 0;
												if (!optimumFound) {
													while ((System.currentTimeMillis() - start) / 1000 <= 1800) {
														totalIter++;
														Individual[] next_population = new Individual[populationSize];
														indexBestChildren = -1;

														for (int j = 0; j < populationSize; j++) {

															Individual[] parents;

															// SELECT PARENTS
															if (selectionParam == 0)
																parents = rouletteWheel();
															else if (selectionParam == 1)
																parents = tournment(2);
															else if (selectionParam == 2)
																parents = tournment(r.nextInt(populationSize - 1) + 2);
															else
																throw new RuntimeException("selection not found");

															// END SELECT PARENTS

															Individual child = null;

															// CROSSOVER
															if (r.nextDouble() <= crossoverProbability) {
																if (crossoverParam == 0)
																	child = onePointCrossover(parents);
																else if (crossoverParam == 1)
																	child = twoPointCrossoverVerI(parents);
																else if (crossoverParam == 2)
																	child = twoPointCrossoverVerII(parents);
																else if (crossoverParam == 3)
																	child = positionBasedCrossover(parents);
																else
																	throw new RuntimeException("crossover not found");
															} else
																child = new Individual(parents[0]);

															// END CROSSOVER

															// MUTATION
															if (r.nextDouble() <= mutationProbability) {
																if (mutationParam == 0)
																	adjacentTwoJobChange(child);
																else if (mutationParam == 1)
																	arbitraryTwoJobChange(child);
																else if (mutationParam == 2)
																	arbitraryThreeJobChange(child);
																else if (mutationParam == 3)
																	shift(child);
																else if (mutationParam == 4)
																	adjacentTwoBatchChange(child);
																else if (mutationParam == 5)
																	arbitraryTwoBatchChange(child);
																else
																	throw new RuntimeException("mutation not found");

															}
															// END MUTATION

															evaluateFitness(child);

															updateUB(child, j);

															if (optimumFound) {
																break;
															}

															localSearch(child);

															updateUB(child, j);

															if (optimumFound) {
																break;
															}

															next_population[j] = child;

															newFitness += child.getFitness();

														}
														if (optimumFound)
															break;

														int childToRemove = r.nextInt(populationSize);

														if (childToRemove == indexBestChildren) {
															childToRemove += 1;
															childToRemove %= populationSize;
														}

														newFitness -= next_population[childToRemove].getFitness();
														newFitness += best.getFitness();

														totalFitness = newFitness;
														newFitness = 0;

														next_population[childToRemove] = best;
														population = next_population;

														if (indexBestChildren != -1
																&& next_population[indexBestChildren]
																		.getFitness() < best.getObjF())
															best = next_population[indexBestChildren];
													}
												}
												totalTime += System.currentTimeMillis() - start;
												totalObjVal += UB;

											}

											// SCRIVERE I RISULTATI SU FILE
											System.out.println(
													"nA = " + nA + " nB = " + nB + " time = " + (totalTime / 50) / 1000
															+ " obj = " + totalObjVal / 50 + " iter" + totalIter / 50);
										}
									}
									closeExcelFile();

								}
	}

	private static void closeExcelFile() {
		if (workBook != null) {
			try {
				workBook.write();
				workBook.close();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (WriteException e) {
				e.printStackTrace();
			}
		}
	}

	private static void createExcelFile() {
		try {
			String path = EXCEL_FILE_LOCATION + "PSize = " + populationSize + " CProb = " + crossoverProbability
					+ " MProb = " + mutationProbability + " Init = " + initParam + " Sel = " + selectionParam
					+ " Cross = " + crossoverParam + " Mut = " + mutationParam + ".xls";
			System.out.println("\n \n");
			System.out.println("-----------------------------------");
			System.out.println("PATH = " + path);
			System.out.println();

			workBook = Workbook.createWorkbook(new File(path));

			// create an Excel sheet
			excelSheet = workBook.createSheet("Lagrangian Results", 0);

			// add header into the Excel sheet
			Label label = new Label(0, 0, "nA");
			excelSheet.addCell(label);

			label = new Label(1, 0, "nB");
			excelSheet.addCell(label);

			label = new Label(2, 0, "f_o");
			excelSheet.addCell(label);

			label = new Label(3, 0, "N. Iter");
			excelSheet.addCell(label);

			label = new Label(4, 0, "time");
			excelSheet.addCell(label);
		} catch (Exception e) {
			throw new RuntimeException("error creating excel file");
		}

	}

	private static void initOutParameters(int populationIndex, int crossoverProbabilityIndex,
			int mutationProbabilityIndex, int initMethodIndex, int selectionMethodIndex, int crossoverMethodIndex,
			int mutationMethodIndex) {

		populationSize = populationSizeArray[populationIndex];
		crossoverProbability = crossoverProbabilityArray[crossoverProbabilityIndex];
		mutationProbability = mutationProbabilityArray[mutationProbabilityIndex];
		initParam = initMethodArray[initMethodIndex];
		selectionParam = selectionMethodArray[selectionMethodIndex];
		crossoverParam = crossoverMethodArray[crossoverMethodIndex];
		mutationParam = mutationMethodArray[mutationMethodIndex];

	}

	private static void updateUB(Individual child, int pos) {
		if (child.getObjF() <= UB) {
			indexBestChildren = pos;
			UB = child.getObjF();
			if (UB <= Math.pow(10, -6))
				optimumFound = true;
		}

	}

	private static void localSearch(Individual child) {
		for (int i = 0; i < nA + nB - 1; i++) {

			for (int j = i + 1; j < nA + nB; j++) {
				double newSumA = child.getSumA();
				double newSumB = child.getSumB();

				for (int k = i; k <= j; k++) {
					if (child.getGenotype()[i] < nA)
						newSumA += p[child.getGenotype()[k]] * w[child.getGenotype()[i]];
					else
						newSumB += p[child.getGenotype()[k]] * w[child.getGenotype()[i]];

					if (child.getGenotype()[j] < nA)
						newSumA -= p[child.getGenotype()[k]] * w[child.getGenotype()[j]];
					else
						newSumB -= p[child.getGenotype()[k]] * w[child.getGenotype()[j]];

					if (k > i && k < j)
						if (child.getGenotype()[k] < nA)
							newSumA += (p[child.getGenotype()[j]] - p[child.getGenotype()[i]])
									* w[child.getGenotype()[k]];
						else
							newSumB += (p[child.getGenotype()[j]] - p[child.getGenotype()[i]])
									* w[child.getGenotype()[k]];

				}
				if (child.getGenotype()[i] < nA)
					newSumA -= p[child.getGenotype()[i]] * w[child.getGenotype()[i]];
				else
					newSumB -= p[child.getGenotype()[i]] * w[child.getGenotype()[i]];

				if (child.getGenotype()[j] < nA)
					newSumA += p[child.getGenotype()[j]] * w[child.getGenotype()[j]];
				else
					newSumB += p[child.getGenotype()[j]] * w[child.getGenotype()[j]];

				if (Math.abs(newSumA / nA - newSumB / nB) < child.getObjF()) {

					child.swap(i, j);
					child.setObjF(Math.abs(newSumA / nA - newSumB / nB));
					if (child.getObjF() <= 0) {
						return;
					}
					child.setFitness(1 / child.getObjF());

					child.setSumA(newSumA);
					child.setSumB(newSumB);

				}

			}
		}
	}

	private static void evaluateFitness(Individual child) {
		double cumulativeTime = 0;
		double sumA = 0;
		double sumB = 0;

		for (int j = 0; j < nA + nB; j++) {
			cumulativeTime += p[child.getGenotype()[j]];

			if (child.getGenotype()[j] < nA)
				sumA += cumulativeTime * w[child.getGenotype()[j]];
			else
				sumB += cumulativeTime * w[child.getGenotype()[j]];
		}

		child.setObjF(Math.abs(sumA / nA - sumB / nB));
		if (child.getObjF() <= 0)
			return;
		child.setFitness(1 / child.getObjF());
		child.setSumA(sumA);
		child.setSumB(sumB);

	}

	private static void arbitraryTwoBatchChange(Individual child) {
		int first = r.nextInt(nA + nB), second = r.nextInt(nA + nB);
		if (second == first) {
			second += 1;
			second %= nA + nB;
		}

		int batchSize = r.nextInt(Math.min(Math.abs(first - second), nA + nB - Math.max(first, second))) + 1;
		for (int i = 0; i < batchSize; i++)
			child.swap(first + i, second + i);

	}

	private static void adjacentTwoBatchChange(Individual child) {
		int start = r.nextInt(nA + nB - 1);
		int batchSize = r.nextInt((nA + nB - start) / 2) + 1;
		for (int i = 0; i < batchSize; i++)
			child.swap(start + i, start + batchSize + i);
	}

	private static void shift(Individual child) {
		int from = r.nextInt(nA + nB), to = r.nextInt(nA + nB);

		int step = from < to ? 1 : -1;

		int element = child.getGenotype()[from];
		for (int index = from; index != to; index += step) {
			child.setJob(index, child.getGenotype()[index + step]);
		}
		child.setJob(to, element);
	}

	private static void arbitraryThreeJobChange(Individual child) {
		int first = r.nextInt(nA + nB), second = r.nextInt(nA + nB), third = r.nextInt(nA + nB);
		child.swap(second, third);
		child.swap(first, second);

	}

	private static void arbitraryTwoJobChange(Individual child) {
		int first = r.nextInt(nA + nB), second = r.nextInt(nA + nB);
		child.swap(first, second);
	}

	private static void adjacentTwoJobChange(Individual child) {
		int point = r.nextInt(nA + nB - 1);
		child.swap(point, point + 1);
	}

	private static Individual positionBasedCrossover(Individual[] parents) {
		Individual child = new Individual(nA + nB);
		boolean child_scheduled[] = new boolean[nA + nB];
		boolean child_position_used[] = new boolean[nA + nB];

		int num_pos = r.nextInt(nA + nB) + 1;

		Collections.shuffle(_1_n, r); // the first numpos elements correspond to the positions

		for (int i = 0; i < num_pos; i++) {
			child.setJob(_1_n.get(i), parents[0].getGenotype()[_1_n.get(i)]);
			child_scheduled[parents[0].getGenotype()[_1_n.get(i)]] = true;
			child_position_used[_1_n.get(i)] = true;
		}

		int index = 0, inserted = 0;
		for (int i = 0; i < nA + nB; i++) {
			if (!child_scheduled[parents[1].getGenotype()[i]]) {
				child_scheduled[parents[1].getGenotype()[i]] = true;
				while (child_position_used[index])
					index++;
				child_position_used[index] = true;
				child.setJob(index, parents[1].getGenotype()[i]);
				inserted += 1;
				if (inserted == nA + nB - num_pos)
					break;
			}
		}
		return child;
	}

	private static Individual twoPointCrossoverVerII(Individual[] parents) {
		Individual child = new Individual(nA + nB);

		int first = r.nextInt(nA + nB) + 1, second = r.nextInt(nA + nB) + 1;

		boolean child_scheduled[] = new boolean[nA + nB];
		for (int j = Math.min(first, second); j < Math.max(first, second); j++) {
			child.setJob(j, parents[0].getGenotype()[j]);
			child_scheduled[parents[0].getGenotype()[j]] = true;
		}

		int index = 0;

		for (int j = 0; j < nA + nB; j++) {
			if (index == Math.min(first, second))
				index = Math.max(first, second);

			if (index == nA + nB)
				break;

			if (!child_scheduled[parents[1].getGenotype()[j]]) {
				child.setJob(index++, parents[1].getGenotype()[j]);
				child_scheduled[parents[1].getGenotype()[j]] = true;
			}

		}

		return child;
	}

	private static Individual twoPointCrossoverVerI(Individual[] parents) {
		Individual child = new Individual(nA + nB);

		int first = r.nextInt(nA + nB) + 1, second = r.nextInt(nA + nB) + 1;
		boolean child_scheduled[] = new boolean[nA + nB];
		for (int j = 0; j < Math.min(first, second); j++) {
			child.setJob(j, parents[0].getGenotype()[j]);
			child_scheduled[parents[0].getGenotype()[j]] = true;
		}
		for (int j = Math.max(first, second); j < nA + nB; j++) {
			child.setJob(j, parents[0].getGenotype()[j]);
			child_scheduled[parents[0].getGenotype()[j]] = true;
		}

		int index = Math.min(first, second);

		for (int j = 0; j < nA + nB; j++) {
			if (!child_scheduled[parents[1].getGenotype()[j]]) {
				child.setJob(index++, parents[1].getGenotype()[j]);
				child_scheduled[parents[1].getGenotype()[j]] = true;
			}
			if (index == Math.max(first, second))
				break;
		}

		return child;
	}

	private static Individual onePointCrossover(Individual[] parents) {
		Individual child = new Individual(nA + nB);
		boolean child_scheduled[] = new boolean[nA + nB];

		int point = r.nextInt(nA + nB - 1) + 1;
		int start, end, index;
		if (r.nextDouble() <= 0.5) {
			start = 0;
			end = point;
			index = point;
		} else {
			start = point;
			end = nA + nB;
			index = 0;
		}

		for (int i = start; i < end; i++) {
			child.setJob(i, parents[0].getGenotype()[i]);
			child_scheduled[parents[0].getGenotype()[i]] = true;
		}

		for (int i = 0; i < nA + nB; i++) {
			if (!child_scheduled[parents[1].getGenotype()[i]]) {
				child.setJob(index++, parents[1].getGenotype()[i]);
				if (index == start || index == nA + nB)
					break;
			}
		}

		return child;
	}

	private static Individual[] tournment(int i) {
		Individual[] toReturn = new Individual[2];
		toReturn[0] = population[r.nextInt(populationSize)];

		toReturn[1] = population[r.nextInt(populationSize)];

		for (int k = 1; k < i; k++) {
			int extracted1 = r.nextInt(populationSize);
			int extracted2 = r.nextInt(populationSize);

			if (population[extracted1].getObjF() < toReturn[0].getObjF())
				toReturn[0] = population[extracted1];

			if (population[extracted2].getObjF() < toReturn[1].getObjF())
				toReturn[1] = population[extracted2];

		}
		return toReturn;

	}

	private static Individual[] rouletteWheel() {

		double firstPoint = r.nextDouble() * totalFitness, secondPoint = r.nextDouble() * totalFitness;
		double cumulative = 0;
		boolean found1 = false, found2 = false;

		Individual toReturn[] = new Individual[2];

		for (int j = 0; j < populationSize && (!found1 || !found2); j++) {
			if (cumulative < firstPoint && firstPoint < cumulative + population[j].getFitness() && !found1) {
				toReturn[0] = population[j];
				found1 = true;
			}
			if (cumulative < secondPoint && secondPoint < cumulative + population[j].getFitness() && !found2) {
				toReturn[1] = population[j];
				found2 = true;
			}
			cumulative += population[j].getFitness();

		}

		return toReturn;
	}

	// NOTA SE NA ED NB NON SONO ENTRAMBI PARI VANNO SCHEDULATI I RIMANENTI JOB
	// (MANCA)
	private static void initBidirectional() {
		totalFitness = 0;
		best = new Individual(nA + nB);

		for (int i = 0; i < populationSize; i++) {
			Collections.shuffle(_1_nA, r);
			Collections.shuffle(nA_n, r);

			Individual current = new Individual(nA + nB);

			double currSumA = 0, currSumB = 0, cumulative = 0, totalSum = pSum;

			for (int j = 0; j < nA / 2 + nB / 2; j++) {
				if (j % 2 == 0 || j >= nB) {
					double case1SumA = currSumA + totalSum * w[_1_nA.get(j + 1)]
							+ (cumulative + p[_1_nA.get(j)]) * w[_1_nA.get(j)];

					double case2SumA = currSumA + totalSum * w[_1_nA.get(j)]
							+ (cumulative + p[_1_nA.get(j + 1)]) * w[_1_nA.get(j + 1)];

					if (Math.abs(case1SumA / nA - currSumB / nB) < Math.abs(case2SumA / nA - currSumB / nB)) {
						current.setJob(j, _1_nA.get(j));
						current.setJob(nA + nB - 1 - j, _1_nA.get(j + 1));

						currSumA = case1SumA;

						cumulative += p[_1_nA.get(j)];
						totalSum -= p[_1_nA.get(j + 1)];
					} else {
						current.setJob(j, _1_nA.get(j + 1));
						current.setJob(nA + nB - 1 - j, _1_nA.get(j));

						currSumA = case2SumA;

						cumulative += p[_1_nA.get(j + 1)];
						totalSum -= p[_1_nA.get(j)];
					}

				} else if (j % 2 == 1) {
					double case1SumB = currSumB + totalSum * w[nA_n.get(j)]
							+ (cumulative + p[nA_n.get(j - 1)]) * w[nA_n.get(j - 1)];

					double case2SumB = currSumB + totalSum * w[nA_n.get(j - 1)]
							+ (cumulative + p[nA_n.get(j)]) * w[nA_n.get(j)];

					if (Math.abs(case1SumB / nB - currSumA / nA) < Math.abs(case2SumB / nB - currSumA / nA)) {
						current.setJob(j, nA_n.get(j - 1));
						current.setJob(nA + nB - 1 - j, nA_n.get(j));

						currSumB = case1SumB;

						cumulative += p[nA_n.get(j - 1)];
						totalSum -= p[nA_n.get(j)];
					} else {
						current.setJob(j, nA_n.get(j));
						current.setJob(nA + nB - 1 - j, nA_n.get(j - 1));

						currSumB = case2SumB;

						cumulative += p[nA_n.get(j)];
						totalSum -= p[nA_n.get(j - 1)];
					}

				}
			}

			current.setObjF(Math.abs(currSumA / nA - currSumB / nB));

			if (current.getObjF() <= UB) {
				UB = current.getObjF();
				best = current;
				if (current.getObjF() <= 0) {
					optimumFound = true;
					return;
				}
			}
			current.setFitness(1 / current.getObjF());
			current.setSumA(currSumA);
			current.setSumB(currSumB);

			population[i] = current;
			totalFitness += current.getFitness();

		}

	}

	private static void initAlternating() {

		totalFitness = 0;
		best = new Individual(nA + nB);

		for (int i = 0; i < populationSize; i++) {
			Collections.shuffle(_1_nA, r);
			Collections.shuffle(nA_n, r);

			Individual current = new Individual(nA + nB);

			double currSumA = 0, currSumB = 0, cumulative = 0;

			for (int j = 0; j < Math.min(nA, nB) * 2; j++) {
				if (j % 2 == 0) {
					current.setJob(j, _1_nA.get(j / 2));
					cumulative += p[_1_nA.get(j / 2)];
					currSumA += cumulative * w[_1_nA.get(j / 2)];
				} else {
					current.setJob(j, nA_n.get(j / 2));
					cumulative += p[nA_n.get(j / 2)];
					currSumB += cumulative * w[nA_n.get(j / 2)];
				}
			}

			for (int j = 0; j < nB - nA; j++) {
				current.setJob(2 * nA + j, nA_n.get(nA + j));
				cumulative += p[nA_n.get(nA + j)];
				currSumB += cumulative * w[nA_n.get(nA + j)];
			}

			current.setObjF(Math.abs(currSumA / nA - currSumB / nB));

			if (current.getObjF() <= UB) {
				UB = current.getObjF();
				best = current;
				if (current.getObjF() <= 0) {
					optimumFound = true;
					return;
				}
			}
			current.setFitness(1 / current.getObjF());
			current.setSumA(currSumA);
			current.setSumB(currSumB);

			population[i] = current;
			totalFitness += current.getFitness();

		}

	}

	private static void initRandom() {
		totalFitness = 0;
		best = new Individual(nA + nB);

		for (int i = 0; i < populationSize; i++) {
			Collections.shuffle(_1_n, r);

			Individual current = new Individual(nA + nB);

			double currSumA = 0, currSumB = 0, cumulative = 0;
			for (int j = 0; j < nA + nB; j++) {
				current.setJob(j, _1_n.get(j));

				cumulative += p[_1_n.get(j)];

				if (_1_n.get(j) < nA)
					currSumA += cumulative * w[_1_n.get(j)];
				else
					currSumB += cumulative * w[_1_n.get(j)];
			}
			current.setObjF(Math.abs(currSumA / nA - currSumB / nB));
			if (current.getObjF() <= UB) {
				UB = current.getObjF();
				best = current;
				if (current.getObjF() <= 0) {
					optimumFound = true;
					return;
				}
			}
			current.setFitness(1 / current.getObjF());
			current.setSumA(currSumA);
			current.setSumB(currSumB);

			population[i] = current;
			totalFitness += current.getFitness();

		}
	}

	private static void initParameters() {
		_1_n = new ArrayList<Integer>();
		_1_nA = new ArrayList<Integer>();
		nA_n = new ArrayList<Integer>();

		for (int j = 0; j < nA; j++) {
			_1_n.add(j);
			_1_nA.add(j);
		}

		for (int j = nA; j < nA + nB; j++) {
			_1_n.add(j);
			nA_n.add(j);
		}

		optimumFound = false;
		population = new Individual[populationSize];

		p = new double[nA + nB];
		w = new double[nA + nB];

		pwSum = 0;
		pSum = 0;

		String instance = scanner_input.nextLine();
		String[] p_w = instance.split(";");

		String[] processingTimeList = p_w[0].split(",");
		String[] weightsList = p_w[1].split(",");

		for (int index = 0; index < nA + nB; index++) {
			p[index] = Integer.valueOf(processingTimeList[index]);
			w[index] = Integer.valueOf(weightsList[index]);

			pSum += p[index];
			pwSum += p[index] * w[index];
		}

		UB = pwSum;

	}
}
