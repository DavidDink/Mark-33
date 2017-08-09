package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import main.Action.Types;

public class Session {
	private HistoryManager hisMan;
	private Container container;
	private TempChanger ac;
	private TempChanger heater;
	
	private float desiredTemp;
	private float desiredHumidity;
	
	private int time;
	
	public Session(Container container, float desiredTemp, float desiredHumidity) {
		this.container = container;
		this.desiredTemp = desiredTemp;
		this.desiredHumidity = desiredHumidity;
		ac = TempChanger.newAirConditioner();
		heater = TempChanger.newHeater();
		hisMan = new HistoryManager(this);
	}

	public Session(float desiredTemp, float desiredHumidity) {
		this(new Container(), desiredTemp, desiredHumidity);
	}
	
	public Session() {
		this(new Container(), 75f, 50f);
	}
	
	public void run() {
		
		desiredTemp = (float)new Random().nextInt(100);
		
		while (!Utils.withinRange(container.getInsideTemp(), desiredTemp, 5f)) {
			// Make time go forward
			++time;
			
			// The action that will (potentially) be taken during this iteration
			Action action;
			
			final float prevInsideTemp = container.getInsideTemp();
			
			// If the outside temperature is colder than the inside temp,
			// or vice versa, it will have an effect on the inside temperature
			// of the container. Let's factor this in.
			final float outsideTempEffect = calculateOutsideTempEffect(
					container.getInsideTemp(), container.getOutsideTemp());

			// Factor in outside temp effect
			container.setInsideTemp(container.getInsideTemp() + outsideTempEffect);
			
			// If it's too cold
			if (container.getInsideTemp() < desiredTemp) {
				ac.setPower(0);
				heater.setPower(desiredTemp - container.getInsideTemp());
				container.setInsideTemp(container.getInsideTemp() + heater.getPower());
				// Set action (turning heater on)
				action = new Action(Types.HEATER_ON, heater.getPower());
			}
			// If it's too hot
			else if (container.getInsideTemp() > desiredTemp) {
				heater.setPower(0);
				final float acPower = desiredTemp - container.getInsideTemp();
				ac.setPower(acPower);
				// Factor in ac power
				container.setInsideTemp(container.getInsideTemp() + ac.getPower());
				// Set action (turning ac on)
				action = new Action(Types.AC_ON, ac.getPower());
			} else {
				// Empty action
				action = new Action();
			}
			
			// If the current inside temp is the same as the one before,
			// the cooler/heater is not strong enough to compete with the outside
			// temp/environment
			if (container.getInsideTemp() == prevInsideTemp) {
				System.err.println("Heater/cooler is not strong enough to compete with " +
						"outside environment.");
				System.exit(0);
			}
			
			// Add this to the history manager
			HistoryManager.Entry entry =
					new HistoryManager.Entry(time, new Container(container), action);
			hisMan.addEntry(entry);
		}
	}
	
	/**
	 * How to calculate how much effect the outside temperature has on the inside
	 * temperature of a container.
	 */
	private float calculateOutsideTempEffect(float inside, float outside) {
		float diff = outside - inside;
		diff /= 20f;
		// Cap diff at 5
		diff = diff > 5f ? 5f : diff < -5f ? -5f : diff;
		if (diff > 10f) {
			System.out.println("Something bad is happening...");
		}
		return diff;
	}
	
	public void saveResultsToFile(String filePath) {
		File file = new File(filePath);
		StringBuilder builder = new StringBuilder();
		for (HistoryManager.Entry e : hisMan.getEntries()) {
			builder.append(e);
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

	public Container getContainer() {
		return container;
	}

	public void setContainer(Container container) {
		this.container = container;
	}

	public float getDesiredTemp() {
		return desiredTemp;
	}

	public void setDesiredTemp(float desiredTemp) {
		this.desiredTemp = desiredTemp;
	}

	public float getDesiredHumidity() {
		return desiredHumidity;
	}

	public void setDesiredHumidity(float desiredHumidity) {
		this.desiredHumidity = desiredHumidity;
	}
}
