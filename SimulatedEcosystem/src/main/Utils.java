package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public final class Utils {
	private Utils() {}
	
	public static boolean withinRange(float val, float desiredVal, float acceptableOffset) {
		return Math.abs(desiredVal - val) <= acceptableOffset;
	}
	
	public static void saveToFile(Session session) {
		String path = "/Users/David/Desktop/";
		String name = "data" + System.currentTimeMillis() + ".csv";
		File file = new File(path + name);
		StringBuilder builder = new StringBuilder();
		// Labels
		builder.append("time,desired temp,inside temp,outside temp,action,comfort penalty,"
				+ "cost,net cost\n");
		
		// Data
		float cost = 0f;
		for (HistoryManager.Entry e : session.getHistoryManager().getEntries()) {
			builder.append(e);
			builder.append(",");
			builder.append(cost += e.getAction().getCost());
			builder.append("\n");
		}
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(file);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		writer.write(builder.toString());
		writer.close();
	}

}
