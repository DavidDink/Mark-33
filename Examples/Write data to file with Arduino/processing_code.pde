/* //<>// //<>//
  Write data from a port to a file. (Processing side).
*/

import processing.serial.*;

// Class that will write data to file
PrintWriter writer;
// The port
Serial port;

void setup() {
  // Print all ports
  printArray(Serial.list());
  // Pick a port
  port = new Serial(this, Serial.list()[1], 9600);
  // Create writer (name = data.txt)
  writer = createWriter("data.txt");
}

void draw() {
  // Get data from port
  String senVal = port.readString();
  if (senVal != null)
    // Write the data to file
    writer.write(senVal);
  // If the data is null
  else {
    writer.flush();
    writer.close();
    exit();
  }
}