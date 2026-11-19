#include <Servo.h>

// ===================== 伺服 & I/O =====================
Servo myservo;
const int VControl_pin = 6;
const int SERVO_PIN = 9;

// ===================== 1) 角度 <-> gap 模型 =====================
// 舊公式 (線性模型，已棄用):
// gap(mm) = M * angle(deg) + B
// const float M = -0.92f;
// const float B = 102.98f;

// 新公式 (三角函數模型):
// gap(cm) = 7 * sin(90° - degree) - 0.7 + 0.044
// gap(cm) = 7 * sin(90° - degree) - 0.656
// gap(mm) = 70 * sin((90 - degree) * PI / 180) - GAP_OFFSET

// Gap 偏移值 (mm)
const float GAP_OFFSET = 13.1f;

// 馬達角度偏移值
// 邏輯角度 90° → 馬達 98°，邏輯角度 45° → 馬達 53°
// motor_angle = logical_angle + MOTOR_OFFSET
const int MOTOR_OFFSET = 2.44;

// 角度範圍（機構限制，邏輯角度）
const int openAngle      = 45;   // 比較開
const int closedAngleMax = 80;   // 閉合最大限制

// 角度 -> gap 轉換 (新公式)
static inline float gapFromAngle(float angleDeg) {
  // gap(cm) = 7 * sin(90° - degree) - 0.656
  // gap(mm) = 70 * sin((90 - degree) * PI / 180) - GAP_OFFSET
  float radians = (90.0f - angleDeg) * PI / 180.0f;
  return 70.0f * sin(radians) - GAP_OFFSET;
}

// gap -> 角度 轉換 (新公式反推)
static inline float angleFromGap(float gapMm) {
  // 反推: gapMm = 70 * sin((90 - angle) * PI / 180) - 6.56
  // sin((90 - angle) * PI / 180) = (gapMm + GAP_OFFSET) / 70
  // (90 - angle) = asin((gapMm + GAP_OFFSET) / 70) * 180 / PI
  // angle = 90 - asin((gapMm + GAP_OFFSET) / 70) * 180 / PI
  float sinVal = (gapMm + GAP_OFFSET) / 70.0f;
  sinVal = constrain(sinVal, -1.0f, 1.0f);  // 防止 asin 超出範圍
  return 90.0f - asin(sinVal) * 180.0f / PI;
}

// 對應可達 gap 範圍 (使用新公式計算)
const float GAP_MAX = 70.0f * sin((90.0f - openAngle) * PI / 180.0f) - GAP_OFFSET;       // openAngle 時的最大 gap
const float GAP_MIN = 70.0f * sin((90.0f - closedAngleMax) * PI / 180.0f) - GAP_OFFSET;  // closedAngleMax 時的最小 gap

// clamp 工具
static inline float clampf(float v, float lo, float hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline float clampGap(float g) {
  return clampf(g, GAP_MIN, GAP_MAX);
}

static inline float clampAngle(float a) {
  return clampf(a, openAngle, closedAngleMax);
}

// ===================== 2) 力模型 F = 117.07(h-g)+27.953 =====================
const float K_FORCE = 117.07f;
const float C_FORCE = 27.953f;

// 給 F,h -> 算 overtravel x = h - g
static inline float overtravelFromForce(float F_g) {
  return (F_g - C_FORCE) / K_FORCE;
}

// 給 h,F -> 算目標 gap g2
static inline float gapFromHAndForce(float h_mm, float F_g) {
  float x = overtravelFromForce(F_g);
  return h_mm - x;
}

// ===================== 3) A4 書本：厚度 -> 重量 -> 夾取力 =====================
const float WEIGHT_PER_MM   = 80.0f;   // g / mm
const float GRIP_FORCE_GAIN = 1.8f;    
const float GRIP_FORCE_MIN  = 150.0f;  // g
const float GRIP_FORCE_MAX  = 1800.0f; // g

static inline float estimateBookWeightFromThickness(float h_mm) {
  return WEIGHT_PER_MM * h_mm;
}

static inline float autoForceFromThickness(float h_mm) {
  float w = estimateBookWeightFromThickness(h_mm);
  float F = w * GRIP_FORCE_GAIN;
  F = clampf(F, GRIP_FORCE_MIN, GRIP_FORCE_MAX);
  return F;
}

// ===================== 當前狀態 =====================
float currentAngle = openAngle;
float currentGap = gapFromAngle(openAngle);

// ===================== 指令處理 =====================
// 支援指令格式：
// G <mm>     - 移動到指定 gap（厚度）
// A <deg>    - 移動到指定角度
// O          - 完全打開
// C          - 夾緊（使用力模型，需先設定厚度 H）
// H <mm>     - 設定書本厚度並夾緊（兩段式：先貼近再施力）
// S          - 查詢當前狀態
// ?          - 說明

float bookThickness = 25.0f;  // 預設書本厚度

void moveToAngle(float angle) {
  angle = clampAngle(angle);
  myservo.write((int)(angle + MOTOR_OFFSET + 0.5f));  // 加上馬達偏移
  currentAngle = angle;
  currentGap = gapFromAngle(angle);
  
  Serial.print(F("OK A="));
  Serial.print(currentAngle, 2);
  Serial.print(F(" G="));
  Serial.println(currentGap, 2);
}

void moveToGap(float gap) {
  gap = clampGap(gap);
  float angle = clampAngle(angleFromGap(gap));
  myservo.write((int)(angle + MOTOR_OFFSET + 0.5f));  // 加上馬達偏移
  currentAngle = angle;
  currentGap = gapFromAngle(angle);  // 用實際角度回算 gap
  
  Serial.print(F("OK A="));
  Serial.print(currentAngle, 2);
  Serial.print(F(" G="));
  Serial.println(currentGap, 2);
}

void openGripper() {
  moveToAngle(openAngle);
  Serial.println(F("OK OPEN"));
}

void gripWithForce(float h_mm) {
  // Stage 1: 先到 gap ≈ h（接觸書本）
  float g1 = clampGap(h_mm);
  float a1 = clampAngle(angleFromGap(g1));
  
  myservo.write((int)(a1 + MOTOR_OFFSET + 0.5f));  // 加上馬達偏移
  Serial.print(F("STAGE1 A="));
  Serial.print(a1, 2);
  Serial.print(F(" G="));
  Serial.println(g1, 2);
  
  delay(500);  // 等待接觸
  
  // Stage 2: 施力
  float F_target = autoForceFromThickness(h_mm);
  float g2 = clampGap(gapFromHAndForce(h_mm, F_target));
  float a2 = clampAngle(angleFromGap(g2));
  
  myservo.write((int)(a2 + MOTOR_OFFSET + 0.5f));  // 加上馬達偏移
  currentAngle = a2;
  currentGap = gapFromAngle(a2);
  
  Serial.print(F("STAGE2 A="));
  Serial.print(currentAngle, 2);
  Serial.print(F(" G="));
  Serial.print(currentGap, 2);
  Serial.print(F(" F="));
  Serial.print(F_target, 1);
  Serial.println(F(" OK"));
}

void printStatus() {
  Serial.println(F("===== 夾爪狀態 ====="));
  Serial.print(F("當前角度 (deg): "));
  Serial.println(currentAngle, 2);
  Serial.print(F("當前 gap (mm):  "));
  Serial.println(currentGap, 2);
  Serial.print(F("設定書本厚度 (mm): "));
  Serial.println(bookThickness, 2);
  Serial.print(F("可達角度範圍 (deg): ["));
  Serial.print(openAngle);
  Serial.print(F(", "));
  Serial.print(closedAngleMax);
  Serial.println(F("]"));
  Serial.print(F("可達 gap 範圍 (mm): ["));
  Serial.print(GAP_MIN, 2);
  Serial.print(F(", "));
  Serial.print(GAP_MAX, 2);
  Serial.println(F("]"));
  Serial.println(F("===================="));
}

void printHelp() {
  Serial.println(F("===== ROS2 夾爪控制指令 ====="));
  Serial.println(F("G <mm>   移動到指定 gap（厚度）"));
  Serial.println(F("A <deg>  移動到指定角度"));
  Serial.println(F("O        完全打開"));
  Serial.println(F("H <mm>   設定書本厚度並執行兩段式夾取"));
  Serial.println(F("S        查詢當前狀態"));
  Serial.println(F("?        顯示此說明"));
  Serial.println(F("T        開啟高壓電"));
  Serial.println(F("Y        關閉高壓電"));
  Serial.println(F("=============================="));
}

void processCommand(String line) {
  line.trim();
  if (line.length() == 0) {
    return;
  }
  
  char cmd = toupper(line[0]);
  String arg = "";
  if (line.length() >= 2) {
    arg = line.substring(1);
    arg.trim();
  }
  
  switch (cmd) {
    case 'G':
      if (arg.length() > 0) {
        float gap = arg.toFloat();
        moveToGap(gap);
      } else {
        Serial.println(F("ERR 格式：G <mm>"));
      }
      break;
      
    case 'A':
      if (arg.length() > 0) {
        float angle = arg.toFloat();
        moveToAngle(angle);
      } else {
        Serial.println(F("ERR 格式：A <deg>"));
      }
      break;
      
    case 'O':
      openGripper();
      break;
      
    case 'H':
      if (arg.length() > 0) {
        bookThickness = arg.toFloat();
        gripWithForce(bookThickness);
      } else {
        Serial.println(F("ERR 格式：H <mm>"));
      }
      break;
      
    case 'T':
      digitalWrite(VControl_pin, HIGH);
      Serial.println(F("V_control = HIGH"));
      break;
    
    case 'Y':
      digitalWrite(VControl_pin, LOW);
      Serial.println(F("V_control = LOW"));
      break;
      
    case 'S':
      printStatus();
      break;
      
    case '?':
      printHelp();
      break;
      
    default:
      Serial.print(F("ERR 未知指令："));
      Serial.println(line);
      break;
  }
}

void readSerialCommands() {
  static String line = "";
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      processCommand(line);
      line = "";
    } else {
      if (line.length() < 64) {
        line += c;
      }
    }
  }
}

// ===================== setup / loop =====================

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  
  pinMode(VControl_pin, OUTPUT);
  digitalWrite(VControl_pin, LOW);  // 預設關閉
  pinMode(3, OUTPUT);
  digitalWrite(3, LOW);  // 預設關閉
  Serial.println(F("=== ROS2 夾爪控制器啟動 ==="));
  printHelp();
  
  myservo.attach(SERVO_PIN);
  
  // 初始化：先打開夾爪
  myservo.write(openAngle);
  currentAngle = openAngle;
  currentGap = gapFromAngle(openAngle);
  
  Serial.print(F("初始狀態：OPEN A="));
  Serial.print(currentAngle, 2);
  Serial.print(F(" G="));
  Serial.println(currentGap, 2);
}

void loop() {
  readSerialCommands();
}
