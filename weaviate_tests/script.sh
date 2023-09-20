until curl --fail -s localhost:8080/v1/.well-known/ready; do
  sleep 1
done
