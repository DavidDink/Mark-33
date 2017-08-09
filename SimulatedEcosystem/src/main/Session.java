package main;

import java.util.Random;

import main.Action.Types;

public class Session {	
	private HistoryManager hisMan;
	private Container container;
	private TempChanger tempChanger;
	
	private float desiredTemp;
	private float desiredHumidity;
	
	private int time;
	
	private Random random;
	
	public Session(Container container, float desiredTemp, float desiredHumidity) {
		this.container = container;
		this.desiredTemp = desiredTemp;
		this.desiredHumidity = desiredHumidity;
		tempChanger = new TempChanger(-2, 2);
		hisMan = new HistoryManager(this);
		random = new Random();
	}

	public Session(float desiredTemp, float desiredHumidity) {
		this(new Container(), desiredTemp, desiredHumidity);
	}
	
	public Session() {
		this(new Container(), 75f, 45f);
	}
	
	public Session(Session other) {
		this(new Container(other.container), other.desiredTemp, other.desiredHumidity);
		tempChanger = new TempChanger(other.tempChanger);
		time = other.time;
	}
	
	public void run(int numTimes) {
		desiredTemp = ComfortManager.IDEAL_TEMP;
		
		while (time < numTimes) {
			if (Utils.withinRange(container.insideTemp(), desiredTemp, 0.1f)) {
				container.setOutsideHumidity(random.nextFloat()*100f);
//				container.setOutsideTemp(random.nextFloat()*100f);
				container.setOutsideTemp(-2000);
			}
			
			// Make time go forward
			++time;
			
			// The action that will (potentially) be taken during this iteration
			Action action;
			// Change in temp
			float dTemp = 0f;
			// If the temperature is in the comfort zone
			final boolean tempInComfortZone =
					ComfortManager.isComfortableTemp(container.insideTemp());
			
			final float prevInsideTemp = container.insideTemp();
			
			// If the outside temperature is colder than the inside temp,
			// or vice versa, it will have an effect on the inside temperature
			// of the container. Let's factor this in.
			final float outsideTempEffect = calculateOutsideTempEffect(
					container.insideTemp(), container.getOutsideTemp());

			// Factor in outside temp effect
			container.setInsideTemp(container.insideTemp() + outsideTempEffect);
			
			if (!tempInComfortZone) {
				final boolean tooCold = container.insideTemp() < desiredTemp;
				final float tempDiff =
						Math.abs(desiredTemp - container.insideTemp()) * (tooCold ? 1f : -1f);
				tempChanger.setPower(tempDiff);
				Action.Types type = tempDiff > 0f ? Types.HEATER_ON : Types.AC_ON;
				final float powerUsed = Math.abs(tempChanger.getPower());
				action = new Action(type, powerUsed);
				dTemp += tempChanger.getPower();
			} else {
				// Empty action
				action = new Action(Action.Types.AC_OFF_HEATER_OFF, 0f);
				tempChanger.setPower(0);
			}
			
			// Apply an error calculation to simulate real-life errors
			dTemp += generateError();
			
			// Update the temperature
			container.setInsideTemp(container.insideTemp() + dTemp);
			
			// If the current inside temp is the same as the one before,
			// the cooler/heater is not strong enough to compete with the outside
			// temp/environment
			if (container.insideTemp() == prevInsideTemp && !tempInComfortZone) {
				System.err.println("Heater/cooler is not strong enough to compete with " +
						"outside environment.");
				System.exit(0);
			}
			
//			// Add this to the history manager
			HistoryManager.Entry entry =
					new HistoryManager.Entry(new Session(this), action);
			hisMan.addEntry(entry);
		}
	}
	
	/**
	 * How to calculate how much effect the outside temperature has on the inside
	 * temperature of a container.
	 */
	private float calculateOutsideTempEffect(float inside, float outside) {
		float diff = outside - inside;
		diff /= 35f;
		// Cap diff at 5
		diff = diff > 5f ? 5f : diff < -5f ? -5f : diff;
		return diff;
	}
	
	private float generateError() {
		return (2 * random.nextFloat() - 1)/10f;
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
	
	public int getCurrentTime() {
		return time;
	}
	
	public HistoryManager getHistoryManager() {
		return hisMan;
	}
}
