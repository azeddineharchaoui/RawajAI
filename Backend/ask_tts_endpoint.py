@app.route('/ask_tts', methods=['POST'])
def ask_question_with_tts():
    """Answer supply chain questions using RAG and return text-to-speech audio"""
    try:
        # Validate request data
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Extract and validate query
        query = data.get('query')
        if not query or not isinstance(query, str):
            return jsonify({"error": "Missing or invalid 'query' parameter"}), 400
            
        # Extract language with default fallback
        language = data.get('language', 'en')
        if language not in ['en', 'fr', 'ar']:
            # Default to English if unsupported language
            language = 'en'
        
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            # Retrieve relevant context
            context = get_rag_context(query)
            if not context:
                context = "No specific context available for this query."
            
            # Generate text response with timeout
            with ThreadPoolExecutor() as executor:
                future = executor.submit(generate_response, query, context, language)
                try:
                    response = future.result(timeout=60)  # 60 second timeout
                except TimeoutError:
                    print("Response generation timed out")
                    timeout_responses = {
                        "en": "I'm sorry, but your request timed out. Please try a simpler question or try again later.",
                        "fr": "Je suis désolé, mais votre demande a expiré. Veuillez essayer une question plus simple ou réessayer plus tard.",
                        "ar": "آسف، لقد انتهت مهلة طلبك. يرجى تجربة سؤال أبسط أو المحاولة مرة أخرى لاحقًا."
                    }
                    response = timeout_responses.get(language, timeout_responses["en"])
            
            # Always generate speech for this endpoint
            speech_file = generate_speech(response, language)
            speech_url = f"/audio/{os.path.basename(speech_file)}" if speech_file else None
            
            # Return both text and speech URL
            return jsonify({
                "query": query,
                "response": response,
                "language": language,
                "speech_url": speech_url,
                "success": True
            })
                
        except Exception as inner_e:
            print(f"Error processing TTS question: {str(inner_e)}")
            
            error_response = "I'm having trouble processing your question right now. Please try again later."
            # Generate speech for error message too
            speech_file = generate_speech(error_response, language)
            speech_url = f"/audio/{os.path.basename(speech_file)}" if speech_file else None
            
            return jsonify({
                "query": query,
                "response": error_response,
                "language": language,
                "speech_url": speech_url,
                "success": False,
                "error": str(inner_e)
            }), 500
            
    except Exception as e:
        print(f"Error with ask_tts request: {str(e)}")
        return jsonify({"error": f"Request error: {str(e)}"}), 400
