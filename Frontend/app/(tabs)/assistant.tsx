"use client"
import AuthenticAdvancedAssistant from "../../components/AuthenticAdvancedAssistant"
import { Platform, StyleSheet } from "react-native"

export default function AssistantScreen() {
  return <AuthenticAdvancedAssistant />
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f7fa",
  },
  header: {
    backgroundColor: "#fff",
    paddingTop: Platform.OS === "ios" ? 50 : 20,
    paddingBottom: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#e1e8ed",
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1a202c",
    textAlign: "center",
  },
  headerSubtitle: {
    fontSize: 14,
    color: "#718096",
    textAlign: "center",
    marginTop: 4,
  },
  messagesContainer: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  messageContainer: {
    marginVertical: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 18,
    maxWidth: "80%",
  },
  userMessage: {
    alignSelf: "flex-end",
    backgroundColor: "#007AFF",
  },
  aiMessage: {
    alignSelf: "flex-start",
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "#e1e8ed",
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  userMessageText: {
    color: "#fff",
  },
  aiMessageText: {
    color: "#1a202c",
  },
  timestamp: {
    fontSize: 12,
    color: "#718096",
    marginTop: 4,
    alignSelf: "flex-end",
  },
})
