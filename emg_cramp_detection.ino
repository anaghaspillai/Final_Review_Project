const int emgPin = A0;
const int motor1 = 9;
const int motor2 = 10;

int threshold = 500;

void setup() {
  pinMode(motor1, OUTPUT);
  pinMode(motor2, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  int emgValue = analogRead(emgPin);
  Serial.println(emgValue);
  delay(200);
  if (emgValue > threshold) {
    // Pulse pattern
    digitalWrite(motor1, HIGH);
    digitalWrite(motor2, HIGH);
    delay(200);

    digitalWrite(motor1, LOW);
    digitalWrite(motor2, LOW);
    delay(200);
  } else {
    digitalWrite(motor1, LOW);
    digitalWrite(motor2, LOW);
  }
}
