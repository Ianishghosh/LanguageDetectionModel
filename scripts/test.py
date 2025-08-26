from live_detect import LiveLanguageDetector

if __name__ == "__main__":
    detector = LiveLanguageDetector()

    test_file = r"C:\Users\anish\OneDrive\Desktop\Language_Transition_Model\datasetss\mini_dataset\hi\common_voice_hi_23795244.mp3"

    result = detector.detect_from_file(test_file)

    print("\n✅ Final Result:")
    print(result)
