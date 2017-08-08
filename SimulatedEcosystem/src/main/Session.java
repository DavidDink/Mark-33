package main;

public class Session {
	private Container container;
	private AirConditioner ac;
	private Heater heater;
	
	private float desiredTemp;
	private float desiredHumidity;
	
	private int time;
	
	public Session(Container container, float desiredTemp, float desiredHumidity) {
		this.container = container;
		this.desiredTemp = desiredTemp;
		this.desiredHumidity = desiredHumidity;
		ac = new AirConditioner();
		heater = new Heater();
	}

	public Session(float desiredTemp, float desiredHumidity) {
		this(new Container(), desiredTemp, desiredHumidity);
	}
	
	public Session() {
		this(new Container(), 75f, 50f);
	}
	
	public void run() {
		
		while (!Utils.withinRange(container.getInsideTemp(), desiredTemp, 5)) {
			++time;
			System.out.println("Before: " + container.getInsideTemp());
			if (container.getInsideTemp() < desiredTemp) {
				container.setInsideTemp(container.getInsideTemp() + heater.getAirTemp());
			}
			else if (container.getInsideTemp() > desiredTemp) {
				container.setInsideTemp(container.getInsideTemp() - ac.getAirTemp());
			}
		}
		
		System.out.println("Final: " + container.getInsideTemp());
		System.out.println("Duration: " + time);
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
